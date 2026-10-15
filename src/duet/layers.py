"""Dual-stream attention and rotary embeddings used by DuET."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import DuETConfig


class ContinuousRotaryEmbedding(nn.Module):
    """RoPE over float positions.

    ``positions`` are already in tau units, i.e. ``p = (t - t_start) / tau``.
    Nothing here quantises time: a note at 10.031 s keeps position 1.55 and is
    never rounded onto a frame index.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def rotate(self, x, positions):
        return apply_rotary(x, *self(positions))

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.unsqueeze(-1).float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return (emb.cos(), emb.sin())


class TimePitchRotaryEmbedding(nn.Module):
    """Independent time/pitch rotations; positions are [seconds, pitch-60, visible]."""

    def __init__(self, dim):
        super().__init__()
        if dim % 4 or dim < 8:
            raise ValueError(
                "Time/pitch RoPE requires head dimension divisible by 4 and >= 8"
            )
        self.axis_dim = dim // 2
        pairs = dim // 4
        self.time_periods = tuple(
            torch.logspace(-1.0, math.log10(64.0), pairs, dtype=torch.float64).tolist()
        )
        self.pitch_periods = tuple(
            torch.logspace(
                math.log10(12.0), math.log10(256.0), pairs, dtype=torch.float64
            ).tolist()
        )

    def rotate(self, x, positions):
        if positions.shape[-1] != 3:
            raise ValueError(
                "Time/pitch coordinates require time, pitch, and visibility"
            )
        outputs = []
        for axis, periods in enumerate((self.time_periods, self.pitch_periods)):
            coordinate = positions[..., axis].float()
            if axis == 1:
                coordinate = coordinate.masked_fill(~positions[..., 2].bool(), 0.0)
            phase = coordinate[..., None] * (
                2
                * torch.pi
                / torch.tensor(periods, device=x.device, dtype=torch.float32)
            )
            phase = torch.cat((phase, phase), -1)
            part = apply_rotary(
                x[..., axis * self.axis_dim : (axis + 1) * self.axis_dim],
                phase.cos(),
                phase.sin(),
            )
            if axis == 1:
                part = part.masked_fill(
                    ~positions[..., 2].bool()[:, None, :, None], 0.0
                )
            outputs.append(part)
        return torch.cat(outputs, -1)


def time_pitch_positions(time, pitch, valid=None):
    visible = torch.ones_like(pitch, dtype=torch.bool)
    if valid is not None:
        visible = visible & valid.bool()
    pitch_coordinate = (pitch.float() - 60).masked_fill(~visible, 0.0)
    return torch.stack((time.float(), pitch_coordinate, visible.float()), -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = (x[..., :half], x[..., half:])
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1).to(dtype=x.dtype)
    sin = sin.unsqueeze(1).to(dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


def _safe_key_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Guarantee every query row keeps at least one key (SDPA would NaN)."""
    if mask is None:
        return None
    empty = ~mask.any(dim=-1, keepdim=True)
    return mask | empty


class RelativeAttention(nn.Module):
    """MHA that applies continuous RoPE with *different* query/key positions."""

    def __init__(self, config: DuETConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.attention_dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        rotary: ContinuousRotaryEmbedding,
        key_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``pair_mask`` (``[B, Lq, Lk]``, True = attendable) adds a per-query
        restriction on top of ``key_mask`` -- e.g. a causal "only strictly
        earlier notes" constraint that a single per-key padding mask cannot
        express. Unlike ``key_mask``, a query row left with no valid key under
        ``pair_mask`` is not silently widened back to "attend to everything":
        it has nothing valid to attend to (e.g. the first note in its causal
        order), so its output is zeroed instead.
        """
        if key_value.shape[1] == 0 or query.shape[1] == 0:
            return torch.zeros_like(query)
        if key_mask is not None:
            key_value = key_value.masked_fill(~key_mask[:, :, None], 0.0)
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key_value))
        v = self._split(self.v_proj(key_value))
        q = rotary.rotate(q, query_pos)
        k = rotary.rotate(k, key_pos)
        b, _, lq, _ = q.shape
        lk = k.shape[2]
        attn_mask = None
        valid_rows = None
        combined = None
        if pair_mask is None and key_mask is not None:
            attn_mask = _safe_key_mask(key_mask)[:, None, None, :]
        elif pair_mask is not None:
            combined = (
                pair_mask if key_mask is None else pair_mask & key_mask[:, None, :]
            )
            valid_rows = combined.any(dim=-1)
            attn_mask = _safe_key_mask(combined)[:, None, :, :].expand(b, 1, lq, lk)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self.o_proj(
            out.transpose(1, 2).contiguous().view(b, lq, self.hidden_size)
        )
        if pair_mask is not None:
            out = out.masked_fill(~valid_rows[:, :, None], 0.0)
        elif key_mask is not None:
            out = out.masked_fill(~key_mask.any(dim=-1)[:, None, None], 0.0)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: DuETConfig):
        super().__init__()
        self.ffn_type = config.ffn_type
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.inner_dropout = nn.Dropout(config.hidden_dropout)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.output_dropout(self.down_proj(self.inner_dropout(hidden)))


class FusionBlock(nn.Module):
    """Self-attention per branch, then bidirectional cross-attention.

    Both cross directions start from the same post-self-attention states so the
    result does not depend on an implicit update order.
    """

    def __init__(self, config: DuETConfig):
        super().__init__()
        self.frame_self = RelativeAttention(config)
        self.note_self = RelativeAttention(config)
        self.frame_cross = RelativeAttention(config)
        self.note_cross = RelativeAttention(config)
        self.frame_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.note_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.frame_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.note_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.frame_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.note_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.frame_ffn = FeedForward(config)
        self.note_ffn = FeedForward(config)

    def forward(
        self,
        frame,
        note,
        frame_pos,
        note_pos,
        rotary,
        frame_mask,
        note_mask,
        note_rotary=None,
        note_coordinates=None,
    ):
        frame = frame + self.frame_self(
            self.frame_norm1(frame),
            self.frame_norm1(frame),
            frame_pos,
            frame_pos,
            rotary,
            frame_mask,
        )
        note = note + self.note_self(
            self.note_norm1(note),
            self.note_norm1(note),
            note_pos if note_coordinates is None else note_coordinates,
            note_pos if note_coordinates is None else note_coordinates,
            rotary if note_rotary is None else note_rotary,
            note_mask,
        )
        fn = self.frame_norm2(frame)
        nn_ = self.note_norm2(note)
        frame = frame + self.frame_cross(
            fn, nn_, frame_pos, note_pos, rotary, note_mask
        )
        note = note + self.note_cross(nn_, fn, note_pos, frame_pos, rotary, frame_mask)
        frame = frame + self.frame_ffn(self.frame_norm3(frame))
        note = note + self.note_ffn(self.note_norm3(note))
        return (frame, note)


class Encoder(nn.Module):
    def __init__(self, config: DuETConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.roll_stem = nn.Sequential(
            nn.Conv1d(config.roll_feature_count, d, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.roll_stem_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.note_pitch_emb = nn.Embedding(128, d // 2)
        self.note_feat_proj = nn.Linear(2, d // 2)
        self.note_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.mask_embedding = nn.Parameter(torch.zeros(d))
        self.rotary = ContinuousRotaryEmbedding(
            config.hidden_size // config.num_attention_heads, base=config.rope_theta
        )
        self.blocks = nn.ModuleList(
            [FusionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.frame_final_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.note_final_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        roll: torch.Tensor,
        frame_times_rel: torch.Tensor,
        note_pitch: torch.Tensor,
        note_onset_rel: torch.Tensor,
        note_duration_sec: torch.Tensor,
        note_velocity: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
        note_masked: torch.Tensor | None = None,
        note_region: torch.Tensor | None = None,
        frame_region: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tau = self.config.tau
        if roll.ndim != 3 or roll.shape[1] != self.config.roll_feature_count:
            raise ValueError(
                "Roll layout does not match checkpoint roll_velocity/pitch configuration"
            )
        if frame_valid is not None:
            roll = roll.masked_fill(~frame_valid[:, None, :], 0.0)
        frame_embedding = roll
        for layer in self.roll_stem:
            frame_embedding = layer(frame_embedding)
            if frame_valid is not None:
                frame_embedding = frame_embedding.masked_fill(
                    ~frame_valid[:, None, :], 0.0
                )
        frame_embedding = frame_embedding.transpose(1, 2)
        frame_embedding = self.roll_stem_norm(frame_embedding)
        if frame_region is not None:
            frame_embedding = frame_embedding + frame_region
        if frame_valid is not None:
            frame_embedding = frame_embedding.masked_fill(~frame_valid[:, :, None], 0.0)
        frame_pos = frame_times_rel.float() / tau
        if frame_valid is None:
            frame_valid = torch.ones_like(frame_pos, dtype=torch.bool)
        pitch_emb = self.note_pitch_emb(note_pitch.clamp(0, 127))
        cont = torch.stack([note_duration_sec, note_velocity], dim=-1).float()
        cont_emb = self.note_feat_proj(cont)
        note_emb = torch.cat([pitch_emb, cont_emb], dim=-1)
        if note_masked is not None:
            note_emb = torch.where(
                note_masked[..., None].bool(), self.mask_embedding, note_emb
            )
        note = self.note_norm(note_emb)
        if note_region is not None:
            note = note + note_region
        note = self.dropout(note)
        note_pos = note_onset_rel.float() / tau
        note_rotary = getattr(self, "note_rotary", None)
        note_coordinates = (
            time_pitch_positions(note_onset_rel, note_pitch, valid=note_mask)
            if note_rotary is not None
            else None
        )
        frame = self.dropout(frame_embedding)
        for block in self.blocks:
            frame, note = block(
                frame,
                note,
                frame_pos,
                note_pos,
                self.rotary,
                frame_valid,
                note_mask,
                note_rotary,
                note_coordinates,
            )
        frame = self.frame_final_norm(frame).masked_fill(~frame_valid[:, :, None], 0.0)
        return {
            "frame": frame,
            "note": self.note_final_norm(note),
            "frame_pos": frame_pos,
            "note_pos": note_pos,
            "frame_valid": frame_valid,
        }

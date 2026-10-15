"""Final note-aligned inference architecture. No training objectives."""

import torch
from torch import nn
from torch.nn import functional as F
from .config import DuETConfig
from .codec import Codec
from .layers import (
    Encoder,
    RelativeAttention,
    FeedForward,
    ContinuousRotaryEmbedding,
    TimePitchRotaryEmbedding,
    time_pitch_positions,
    apply_rotary,
)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.self_attention = RelativeAttention(config)
        self.cross_attention = nn.MultiheadAttention(
            d,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def cross(
        self,
        query,
        memory,
        memory_valid,
        rotary,
        positions,
        cache=None,
        key_positions=None,
    ):
        if key_positions is None:
            raise ValueError("Cross-attention requires explicit memory positions")
        a = self.cross_attention
        weights = a.in_proj_weight.chunk(3)
        biases = a.in_proj_bias.chunk(3)
        split = self.self_attention._split
        q = rotary.rotate(split(F.linear(query, weights[0], biases[0])), positions)
        if cache is not None and "cross_k" in cache:
            k, v = (cache["cross_k"], cache["cross_v"])
        else:
            k = rotary.rotate(
                split(F.linear(memory, weights[1], biases[1])), key_positions
            )
            v = split(F.linear(memory, weights[2], biases[2]))
            if cache is not None:
                cache.update(cross_k=k, cross_v=v)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=memory_valid[:, None, None, :],
            dropout_p=a.dropout if self.training else 0.0,
        )
        return a.out_proj(y.transpose(1, 2).reshape_as(query))

    def forward(
        self,
        x,
        memory,
        valid,
        memory_valid,
        rotary,
        query_positions,
        key_positions,
        cross_rotary=None,
    ):
        pos = (
            torch.arange(x.shape[1], device=x.device)
            .float()[None]
            .expand(x.shape[0], -1)
        )
        causal = torch.ones(
            x.shape[1], x.shape[1], device=x.device, dtype=torch.bool
        ).tril()[None]
        q = self.norms[0](x)
        x = x + self.dropout(self.self_attention(q, q, pos, pos, rotary, valid, causal))
        x = x + self.dropout(
            self.cross(
                self.norms[1](x),
                memory,
                memory_valid,
                cross_rotary if cross_rotary is not None else rotary,
                query_positions,
                key_positions=key_positions,
            )
        )
        return (x + self.ffn(self.norms[2](x))).masked_fill(~valid[..., None], 0.0)

    def step(
        self,
        x,
        memory,
        memory_valid,
        rotary,
        cache,
        position,
        query_positions,
        key_positions,
        cross_rotary=None,
    ):
        a = self.self_attention
        q = self.norms[0](x)
        pos = torch.full(x.shape[:2], float(position), device=x.device)
        qk = rotary(pos)
        query = apply_rotary(a._split(a.q_proj(q)), *qk)
        key = apply_rotary(a._split(a.k_proj(q)), *qk)
        value = a._split(a.v_proj(q))
        if "self_k" in cache:
            key = torch.cat((cache["self_k"], key), 2)
            value = torch.cat((cache["self_v"], value), 2)
        cache.update(self_k=key, self_v=value)
        y = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        x = x + a.o_proj(y.transpose(1, 2).reshape_as(x))
        x = x + self.cross(
            self.norms[1](x),
            memory,
            memory_valid,
            cross_rotary if cross_rotary is not None else rotary,
            query_positions,
            cache,
            key_positions,
        )
        return x + self.ffn(self.norms[2](x))


class DuETModel(nn.Module):
    history_sec = 8.0
    prefix_bars = 0
    note_time_pitch_rope = True
    decoder_input_pitch = True

    def __init__(self):
        super().__init__()
        self.config = DuETConfig()
        c = self.config
        d = c.hidden_size
        self.codec = Codec()
        self.encoder = Encoder(c)
        self.encoder.note_rotary = TimePitchRotaryEmbedding(d // c.num_attention_heads)
        self.note_rotary = TimePitchRotaryEmbedding(d // c.num_attention_heads)
        self.rotary = ContinuousRotaryEmbedding(
            d // c.num_attention_heads, base=c.rope_theta
        )
        self.note_region_embedding = nn.Embedding(2, d)
        self.frame_region_embedding = nn.Embedding(2, d)
        self.decoder_region_embedding = nn.Embedding(2, d)
        self.memory_norm = nn.LayerNorm(d)
        self.frame_memory_norm = nn.LayerNorm(d)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d))
        self.null_frame_memory = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([DecoderBlock(c) for _ in range(8)])
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n in self.codec.sizes])
        self.heads = nn.ModuleList([nn.Linear(d, n) for n in self.codec.sizes])
        self.bos_embedding = nn.Parameter(torch.zeros(d))
        self.current_pitch_embedding = nn.Embedding(129, d)
        self.final_norm = nn.LayerNorm(d)
        self.pitch_head = nn.Linear(d, 128)
        self.beat_head = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, 3, 1)
        )
        self.beat_index_head = nn.Linear(d, 64)
        self.frame_phase_cos_head = nn.Linear(d, 1)
        self.frame_phase_sin_head = nn.Linear(d, 1)

    def encode(self, batch):
        fields = (
            "roll",
            "frame_times_rel",
            "note_pitch",
            "note_onset_rel",
            "note_duration_sec",
            "note_velocity",
            "frame_valid",
            "note_mask",
            "note_masked",
        )
        inputs = {k: batch[k] for k in fields if k in batch}
        inputs["note_region"] = self.note_region_embedding(
            batch["note_is_context"].long()
        )
        inputs["frame_region"] = self.frame_region_embedding(
            batch["frame_is_context"].long()
        )
        out = self.encoder(**inputs)
        selected = out["note"]
        valid = batch.get(
            "note_mask",
            torch.ones(selected.shape[:2], device=selected.device, dtype=torch.bool),
        )
        memory = self.memory_norm(selected)
        memory = torch.cat(
            [memory, self.null_memory.expand(len(selected), -1, -1)], dim=1
        )
        valid = torch.cat([valid, ~valid.any(-1, keepdim=True)], dim=1)
        frame = self.frame_memory_norm(out["frame"])
        frame_valid = out["frame_valid"]
        frame = torch.cat(
            [frame, self.null_frame_memory.expand(len(frame), -1, -1)], dim=1
        )
        frame_valid = torch.cat(
            [frame_valid, ~frame_valid.any(-1, keepdim=True)], dim=1
        )
        return (
            {"note": memory, "frame": frame},
            {"note": valid, "frame": frame_valid},
            out,
        )

    def positions(self, batch, memory):
        slots = batch["query_note_slots"]
        frame = torch.cat(
            (
                batch["frame_times_rel"] / self.config.tau,
                batch["frame_times_rel"].new_zeros(len(slots), 1),
            ),
            1,
        )
        query_time = (batch["note_onset_rel"] / self.config.tau).gather(1, slots)
        coordinates = time_pitch_positions(
            batch["note_onset_rel"], batch["note_pitch"], valid=batch["note_mask"]
        )
        query = coordinates.gather(1, slots[..., None].expand(-1, -1, 3))
        keys = torch.cat((coordinates, coordinates.new_zeros(len(slots), 1, 3)), 1)
        return dict(note=(query, keys), frame=(query_time, frame))

    def decoder_stream(self, index):
        return "note" if index % 2 == 0 else "frame"

    def cross_rotary(self, stream):
        return self.note_rotary if stream == "note" else self.rotary

    def embed(self, ids, known, context, first=False):
        x = sum(
            (
                e(ids[..., i]) * known[..., i, None]
                for i, e in enumerate(self.embeddings)
            )
        )
        if first:
            x[:, 0] = self.bos_embedding
        return x + self.decoder_region_embedding(context.long())

    def current_pitch(self, batch):
        slots = batch["query_note_slots"]
        pitch = batch["note_pitch"].gather(1, slots)
        masked = batch["note_masked"].gather(1, slots).bool()
        valid = batch["note_mask"].gather(1, slots).bool()
        if "decoder_attention_mask" in batch:
            valid = valid & batch["decoder_attention_mask"]
        ids = torch.where(masked, 128, pitch).masked_fill(~valid, 0).long()
        return self.current_pitch_embedding(ids).masked_fill(~valid[..., None], 0.0)

    def decode_aligned(self, batch, memory, valid, return_hidden=False):
        x = self.embed(
            batch["decoder_input_ids"],
            batch["decoder_input_field_mask"],
            batch["decoder_context"],
            first=True,
        )
        if self.decoder_input_pitch:
            x = x + self.current_pitch(batch)
        pos = self.positions(batch, memory)
        for i, block in enumerate(self.blocks):
            stream = self.decoder_stream(i)
            q, k = pos[stream]
            x = block(
                x,
                memory[stream],
                batch["decoder_attention_mask"],
                valid[stream],
                self.rotary,
                q,
                k,
                self.cross_rotary(stream),
            )
        x = self.final_norm(x)
        logits = [head(x) for head in self.heads]
        return (logits, x) if return_hidden else logits

    def constrain(self, logits, anchor):
        scores = [x[0, -1].float().clone() for x in logits]
        scores[0][
            torch.arange(len(scores[0]), device=scores[0].device)
            - self.codec.bar_offset
            + anchor
            < 0
        ] = -torch.inf
        lengths = torch.as_tensor(
            self.codec.tick_values["bar_length"], device=scores[0].device
        )
        scores[5][lengths <= 0] = -torch.inf
        row = torch.stack([v.argmax() for v in scores])
        length = lengths[row[5]]
        fractions = torch.as_tensor(self.codec.fractions, device=row.device)
        joint = scores[1][:, None] + scores[2][None, :]
        positions = (
            torch.arange(64, device=row.device)[:, None] * 480 + fractions[None, :]
        )
        joint = joint.masked_fill(positions >= length, -torch.inf)
        best = joint.flatten().argmax()
        row[1] = best // len(fractions)
        row[2] = best % len(fractions)
        if row[10]:
            row[3] = 0
        else:
            scores[3][0] = -torch.inf
            row[3] = scores[3].argmax()
        return row

    @torch.no_grad()
    def generate_rows(
        self, batch, *, prefix_rows=None, prefix_known=None, anchor=0, use_cache=True
    ):
        if self.training or len(batch["note_pitch"]) != 1:
            raise ValueError("Aligned generation requires eval and batch size 1")
        slots = batch["query_note_slots"]
        n = slots.shape[1]
        if (
            slots.ndim != 2
            or (slots < 0).any()
            or (slots >= batch["note_pitch"].shape[1]).any()
        ):
            raise ValueError("Invalid query note slots")
        if (
            not batch["note_mask"].gather(1, slots).all()
            or (slots[:, 1:] <= slots[:, :-1]).any()
        ):
            raise ValueError("Queries must address distinct valid input notes in order")
        prefix_rows = (
            torch.empty((0, self.codec.width), dtype=torch.long, device=slots.device)
            if prefix_rows is None
            else prefix_rows
        )
        if len(prefix_rows) > n:
            raise ValueError("Prefix exceeds query count")
        if prefix_known is None:
            prefix_known = torch.ones_like(prefix_rows, dtype=torch.bool)
        memory, valid, _ = self.encode(batch)
        pos = self.positions(batch, memory)
        cache = [{} for _ in self.blocks]
        pitch = self.current_pitch(batch) if self.decoder_input_pitch else None
        rows = []
        knowns = []
        for i in range(n):
            if i == 0:
                ids = torch.zeros(
                    (1, 1, self.codec.width), dtype=torch.long, device=slots.device
                )
                known = torch.zeros_like(ids, dtype=torch.bool)
            else:
                ids = rows[-1][None, None]
                known = knowns[-1][None, None]
            context = torch.full(
                (1, 1), i < len(prefix_rows), device=slots.device, dtype=torch.bool
            )
            if use_cache:
                x = self.embed(ids, known, context, first=i == 0)
                if pitch is not None:
                    x = x + pitch[:, i : i + 1]
                for j, block in enumerate(self.blocks):
                    stream = self.decoder_stream(j)
                    q, k = pos[stream]
                    x = block.step(
                        x,
                        memory[stream],
                        valid[stream],
                        self.rotary,
                        cache[j],
                        i,
                        q[:, i : i + 1],
                        k,
                        self.cross_rotary(stream),
                    )
                hidden = self.final_norm(x)
                logits = [head(hidden) for head in self.heads]
            else:
                previous = torch.stack([torch.zeros_like(ids[0, 0]), *rows])[None]
                prev_known = torch.stack([torch.zeros_like(known[0, 0]), *knowns])[None]
                b = dict(
                    batch,
                    decoder_input_ids=previous,
                    decoder_input_field_mask=prev_known,
                    decoder_attention_mask=torch.ones(
                        1, i + 1, device=slots.device, dtype=torch.bool
                    ),
                    decoder_context=torch.arange(i + 1, device=slots.device)[None]
                    < len(prefix_rows),
                    query_note_slots=slots[:, : i + 1],
                )
                logits, hidden = self.decode_aligned(
                    b, memory, valid, return_hidden=True
                )
            row = (
                prefix_rows[i].clone()
                if i < len(prefix_rows)
                else self.constrain(logits, anchor)
            )
            rows.append(row)
            knowns.append(
                prefix_known[i]
                if i < len(prefix_rows)
                else torch.ones_like(row, dtype=torch.bool)
            )
        result = (
            torch.stack(rows)
            if rows
            else torch.empty(0, self.codec.width, dtype=torch.long, device=slots.device)
        )
        return result

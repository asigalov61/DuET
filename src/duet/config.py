"""Fixed architecture and inference geometry of the epoch83 release."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DuETConfig:
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 1408
    ffn_type: str = "swiglu"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tau: float = 0.02
    pitch_min: int = 21
    pitch_max: int = 108
    roll_velocity: bool = True

    @property
    def roll_feature_count(self):
        return 354

    def to_dict(self):
        return asdict(self)


BUNDLE_FORMAT = "duet-epoch83-v1"
CORE_SECONDS = 32.0
HISTORY_SECONDS = 8.0
MAX_NOTES = 4096

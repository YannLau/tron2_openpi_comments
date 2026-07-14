"""RTC configuration classes.

Based on LeRobot's RTC implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/configuration_rtc.py
"""

from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    """Prefix attention weight schedule for RTC guidance.

    Determines how the weight decays from the prefix (previously executed steps)
    to the suffix (future steps) in the action chunk.

    - ZEROS: Hard switch - weight=1 for prefix, weight=0 for suffix
    - ONES: Full weight up to execution_horizon, then 0
    - LINEAR: Linear decay from 1 to 0 between inference_delay and execution_horizon
    - EXP: Exponential decay (linear * exp modulation) between inference_delay and execution_horizon
    """

    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Configuration for Real-Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Whether RTC is enabled
    enabled: bool = False

    # Prefix attention schedule: how weights decay from prefix to suffix
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR

    # Maximum guidance weight (clamped to prevent divergence)
    max_guidance_weight: float = 10.0

    # Number of steps to execute before requesting a new action chunk
    execution_horizon: int = 10

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if isinstance(self.prefix_attention_schedule, str):
            self.prefix_attention_schedule = RTCAttentionSchedule(self.prefix_attention_schedule)

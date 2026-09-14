from collections.abc import Mapping
import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

DEFAULT_TRON2_STATE_DIM = 16
DEFAULT_TRON2_ACTION_DIM = 16
MAX_TRON2_PHYSICAL_DIM = 32


def validate_tron2_dimensions(state_dim: int, action_dim: int) -> None:
    if state_dim != action_dim or not 1 <= state_dim <= MAX_TRON2_PHYSICAL_DIM:
        raise ValueError(
            "TRON2 physical dimensions must be a matching pair between 1 and "
            f"{MAX_TRON2_PHYSICAL_DIM}, "
            f"got state_dim={state_dim}, action_dim={action_dim}"
        )


def _norm_stats_value(stats: object, field: str) -> object | None:
    value = getattr(stats, field, None)
    if value is None and isinstance(stats, Mapping):
        value = stats.get(field)
    return value


def validate_tron2_norm_stats(
    norm_stats: Mapping[str, object] | None,
    *,
    state_dim: int,
    action_dim: int,
) -> None:
    """Validate physical dimensions encoded by norm stats when stats are available."""
    validate_tron2_dimensions(state_dim, action_dim)
    if norm_stats is None:
        return
    for key, expected_dim in (("state", state_dim), ("actions", action_dim)):
        if key not in norm_stats:
            raise ValueError(f"TRON2 norm stats are missing required key {key!r}")
        stats = norm_stats[key]
        for field in ("mean", "std", "q01", "q99"):
            value = _norm_stats_value(stats, field)
            if value is None:
                if field in ("mean", "std"):
                    raise ValueError(f"TRON2 {key} norm stats do not contain a {field} array")
                continue
            shape = np.asarray(value).shape
            if not shape:
                raise ValueError(f"TRON2 {key} norm stats {field} must have at least one dimension")
            actual_dim = shape[-1]
            if actual_dim != expected_dim:
                label = f"{key} norm dimension" if field == "mean" else f"{key} norm {field} dimension"
                raise ValueError(f"TRON2 {label} mismatch: expected {expected_dim}, got {actual_dim}")


def make_tron2_example(state_dim: int = DEFAULT_TRON2_STATE_DIM) -> dict:
    """Creates a random input example for the TRON2 policy."""
    validate_tron2_dimensions(state_dim, state_dim)
    return {
        "state": np.ones((state_dim,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class Tron2Inputs(transforms.DataTransformFn):
    """Inputs for the TRON2 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [state_dim]
    - actions: [action_horizon, state_dim]
    """

    # TRON2 data is normally already in the pi runtime joint space. This compatibility
    # path is only for legacy datasets that need conversion before training/inference.
    adapt_to_pi: bool = False
    state_dim: int = DEFAULT_TRON2_STATE_DIM
    action_dim: int = DEFAULT_TRON2_ACTION_DIM

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __post_init__(self):
        validate_tron2_dimensions(self.state_dim, self.action_dim)

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"TRON2 state dimension mismatch: expected {self.state_dim}, got {state.shape[-1]}")
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"TRON2 action dimension mismatch: expected {self.action_dim}, got {actions.shape[-1]}"
                )

        data = _decode_tron2(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "subtask" in data:
            inputs["subtask"] = data["subtask"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Tron2Outputs(transforms.DataTransformFn):
    """Outputs for the TRON2 policy."""

    # TRON2 data is normally already in the pi runtime joint space. This compatibility
    # path is only for legacy datasets that need conversion before training/inference.
    adapt_to_pi: bool = False
    action_dim: int = DEFAULT_TRON2_ACTION_DIM

    def __post_init__(self):
        validate_tron2_dimensions(self.action_dim, self.action_dim)

    def __call__(self, data: dict) -> dict:
        model_actions = np.asarray(data["actions"])
        if model_actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"TRON2 model actions must have at least {self.action_dim} dimensions, got {model_actions.shape[-1]}"
            )
        actions = model_actions[..., : self.action_dim]
        outputs = {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
        if "subtask" in data:
            outputs["subtask"] = data["subtask"]
        return outputs


def _legacy_joint_flip_mask(dim: int) -> np.ndarray:
    """Used by the optional legacy joint-space compatibility path."""
    mask = np.ones((dim,), dtype=np.int64)
    base_mask = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
    mask[: min(dim, base_mask.shape[0])] = base_mask[:dim]
    return mask


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Legacy gripper data may be stored in a linear space. This reverses that
    # transformation to stay consistent with pi0's angular gripper convention.
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and the legacy calibration uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the pi0 gripper convention to the legacy gripper range.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_tron2(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _legacy_gripper_indices(dim: int) -> list[int]:
    return [index for index in (6, 13) if index < dim]


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _legacy_joint_flip_mask(state.shape[-1]) * state
        indices = _legacy_gripper_indices(state.shape[-1])
        if indices:
            state[..., indices] = _gripper_to_angular(state[..., indices])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _legacy_joint_flip_mask(actions.shape[-1]) * actions
        indices = _legacy_gripper_indices(actions.shape[-1])
        if indices:
            actions[..., indices] = _gripper_from_angular(actions[..., indices])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _legacy_joint_flip_mask(actions.shape[-1]) * actions
        indices = _legacy_gripper_indices(actions.shape[-1])
        if indices:
            actions[..., indices] = _gripper_from_angular_inv(actions[..., indices])
    return actions

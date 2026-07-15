# TRON2 OpenPI Deployment

[中文文档](README_CN.md)

`tron2_openpi` is a TRON2 deployment-focused derivative of the OpenPI project.
It keeps the OpenPI policy-serving and pi0/pi0.5 model stack, then adds TRON2
policy transforms, deployment configuration templates, and real-robot client
examples for TRON2.

This repository is meant to be used together with the sibling `tron2_env`
runtime package. It is an integration and deployment example, not a complete
release of private checkpoints, datasets, low-level robot SDKs, or local
deployment profiles.

## What This Repository Provides

- pi0.5 policy serving through `scripts/serve_policy.py`.
- TRON2 policy input/output transforms in `src/openpi/policies/tron2_policy.py`.
- TRON2 training/deployment config registrations in `src/openpi/training/config.py`.
- TRON2 robot clients in `examples/tron2/`.
- Public deployment templates in `configs/deploy/`.
- YAML-driven TRON2 task training via `scripts/train_tron2_task.py` and
  `configs/train/tron2_tasks/example.yaml`.
- Optional bridge observation mode for images and state from TRON2 Bridge.
- Optional legacy RealSense observation mode for directly attached cameras.
- RTC deployment client with warmup, observation-timeout recovery, queue
  diagnostics, and optional action smoothing.
- OpenPI client package under `packages/openpi-client/`.

## What Is Not Included

- Model weights and checkpoint directories.
- Training datasets, evaluation datasets, logs, or benchmark results.
- Private `.local.yaml` deployment files.
- Credentials, real camera serial numbers, customer data, or private local
  deployment profiles.
- The undeveloped low-level robot transport.
- A safety certification for unattended robot operation.

## Repository Layout

```text
parent-directory/
├── tron2_openpi/
│   ├── configs/
│   │   ├── deploy/
│   │   │   ├── candy.yaml
│   │   │   ├── tron2_deploy.example.yaml
│   │   │   └── tron2_deploy.example_CN.yaml
│   │   └── train/
│   │       └── tron2_tasks/
│   │           └── example.yaml
│   ├── examples/
│   │   └── tron2/
│   │       ├── deploy_config.py
│   │       ├── pi_client.py
│   │       └── pi_client_rtc.py
│   ├── packages/
│   │   └── openpi-client/
│   ├── scripts/
│   │   ├── cloud_train_entrypoint_portable.sh
│   │   ├── compute_norm_stats.py
│   │   ├── serve_policy.py
│   │   └── train_tron2_task.py
│   └── src/
│       └── openpi/
└── tron2_env/
```

Keep `tron2_openpi/` and `tron2_env/` side by side. The TRON2 client adds the
sibling `../tron2_env/src` path at startup so it can import the runtime package.
Recorded-action replay utilities live in `../tron2_env/examples/replay_data.py`.

## Requirements

- Ubuntu 22.04 is the primary deployment target.
- Python 3.11 or newer.
- `uv` for Python dependency management.
- NVIDIA GPU and CUDA-compatible JAX environment for policy inference.
- A reachable TRON2 WebSocket robot controller.
- TRON2 Bridge access when using `client.observation_source: bridge`.
- Intel RealSense cameras and local camera access when using
  `client.observation_source: legacy`.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

Clone this repository and the independent TRON2 runtime repository as siblings:

```bash
git clone https://github.com/limx-tron2/tron2_openpi.git
git clone https://github.com/limx-tron2/tron2_env.git

cd tron2_openpi
uv sync
uv pip install -e .

cd ../tron2_env
python -m pip install -e ".[bridge,openpi]"
```

Verify that the sibling runtime can be imported:

```bash
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run python -c "import tron2_env; print(tron2_env.__file__)"
```

The printed path should point to `../tron2_env/src/tron2_env/__init__.py`.

## Deployment Configuration

Use the public templates as starting points:

- Public Candy task example: `configs/deploy/candy.yaml`
- English template: `configs/deploy/tron2_deploy.example.yaml`
- Chinese template: `configs/deploy/tron2_deploy.example_CN.yaml`

`configs/deploy/candy.yaml` is a public example profile for the Candy task. It may
include robot LAN IP addresses and task-specific example values. Before running
it on your own robot, verify the checkpoint path, robot address, Bridge address,
initial pose, and safety procedures for your deployment.

Create a private local profile and edit only the local file:

```bash
cp configs/deploy/tron2_deploy.example.yaml configs/deploy/tron2_deploy.local.yaml
```

Do not commit `.local.yaml` files. They are for private paths, robot addresses,
Bridge hosts, and camera serial numbers.

Minimal local fields:

```yaml
policy:
  config: TASK_CONFIG_NAME
  repo_id: DATASET_REPO_ID
  checkpoint_dir: /path/to/checkpoints/TASK_CONFIG_NAME/experiment/step
  default_prompt: TASK_PROMPT

robot:
  ip: ROBOT_IP

client:
  observation_source: bridge
  max_steps: null
```

Important fields:

| Field | Description |
| --- | --- |
| `policy.config` | Training config name registered in `src/openpi/training/config.py`. |
| `policy.repo_id` | Asset directory name used to load normalization statistics. |
| `policy.checkpoint_dir` | Path to the trained checkpoint step directory. |
| `policy.default_prompt` | Default language instruction when the client does not pass `--prompt`. |
| `policy.record` | Saves raw policy inputs/outputs for debugging when `true`. |
| `policy.action_horizon` | Optional inference action chunk length override. |
| `policy.state_dim` | Optional state/action output dimension override for TRON2. |
| `policy.use_delta_joint_actions` | Optional override for delta-action transforms. |
| `server.host` / `server.port` | Policy server listen address. |
| `client.policy_host` / `client.policy_port` | Policy server address from the client process. |
| `client.observation_source` | `bridge` or `legacy`. |
| `client.state_dim` | `16` for arms+grippers, `18` when head joints are included. |
| `client.fps` | Policy action playback rate. |
| `client.publish_rate` | Background ServoJ command publication rate. |
| `client.max_steps` | Number of policy chunks to run; `null` means run until stopped. |
| `client.rtc_enabled` | Use `pi_client_rtc.py` when `true`; use `pi_client.py` when `false`. |
| `client.duration` | RTC runtime in seconds; `0` means run until stopped. |
| `client.execution_horizon` / `client.delay` | RTC `s` and initial `d` timing values. |
| `client.rtc_guidance_enabled` | Enables inference-time RTC VJP guidance. |
| `client.trained_rtc_mode` | Uses training-time RTC conditioning when the checkpoint was trained for it. |
| `robot.ip` / `robot.port` | TRON2 WebSocket controller address. |
| `bridge.host` | TRON2 Bridge WebSocket host for bridge observations. |
| `camera.serial_to_name` | RealSense serial-to-policy-camera-name mapping for legacy mode. |

`policy.repo_id` must match the asset directory inside the checkpoint:

```text
checkpoint_dir/assets/<policy.repo_id>/norm_stats.json
```

Example TRON2 config names currently registered in the code:

| `policy.config` | `policy.repo_id` |
| --- | --- |
| `pi05_tron2_alarm` | `alarm` |
| `pi05_tron2_Banana` | `banana` |
| `pi05_tron2_cabinet` | `cabinet` |
| `pi05_tron2_Candy` | `candy` |
| `pi05_tron2_Chess` | `chess` |
| `pi05_tron2_Cloth` | `cloth` |
| `pi05_tron2_Drawer` | `drawer` |
| `pi05_tron2_Duck` | `duck` |
| `pi05_tron2_SortFruit` | `sort` |

## Observation Modes

Bridge mode uses TRON2 Bridge for images and, by default, bridge-aligned state:

```yaml
client:
  observation_source: bridge

bridge:
  host: wss://BRIDGE_HOST
  state_source: bridge
```

Legacy mode uses local RealSense cameras and robot WebSocket state:

```yaml
client:
  observation_source: legacy

camera:
  serial_to_name:
    HEAD_CAMERA_SERIAL: cam_high
    LEFT_WRIST_CAMERA_SERIAL: cam_left_wrist
    RIGHT_WRIST_CAMERA_SERIAL: cam_right_wrist
```

The TRON2 policy expects these image keys:

- `cam_high`
- `cam_left_wrist`
- `cam_right_wrist`

## Run Policy Serving

Start the policy server:

```bash
uv run scripts/serve_policy.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

Start the TRON2 client in another terminal:

**Note: the robot should be in the initial state after L1+X, then switched to
advanced developer mode. After the client starts, the robot will spread both
arms sideways and then move them forward. If the robot is not in the initial
state, it may lift both arms directly; keep the front workspace clear.**

```bash
uv run python examples/tron2/pi_client.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

Override the prompt for one run:

```bash
uv run python examples/tron2/pi_client.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml \
  --prompt="put the object into the drawer"
```

Stop the client manually when `client.max_steps` is `null`.

## RTC Deployment

RTC uses the same server command. The server detects whether the loaded model
supports RTC and publishes `rtc_enabled` plus `action_horizon` in websocket
metadata. The client supplies runtime timing from the YAML.

Set these fields in your private local YAML:

```yaml
client:
  rtc_enabled: true
  duration: 120
  fps: 30
  execution_horizon: 10
  delay: 2
  rtc_guidance_enabled: true
  rtc_guidance_weight: 10.0
  trained_rtc_mode: false
```

Then run:

```bash
uv run python examples/tron2/pi_client_rtc.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

The RTC client warms up the model, seeds the action queue, retries short
observation timeouts without reusing stale observations, records queue
diagnostics, and can apply optional client-side action smoothing through
`client.rtc_action_postprocess`.

## Training A New TRON2 Task

For public task configs, prefer the YAML entry point instead of editing
`src/openpi/training/config.py`:

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
```

Edit `configs/train/tron2_tasks/my_task.yaml`, then point LeRobot at your dataset
root. If `repo_id: my_dataset`, the dataset should normally be available under
`$HF_LEROBOT_HOME/my_dataset/` with `data/` and `meta/` subdirectories:

```bash
export HF_LEROBOT_HOME=/path/to/datasets
```

Compute normalization statistics before the first training run:

```bash
uv run scripts/compute_norm_stats.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

Then start training:

```bash
uv run scripts/train_tron2_task.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

For a one-command local/container workflow that mirrors the internal two-stage
training flow, use the portable entrypoint. It computes normalization statistics
first unless `--skip-norm` is passed, then launches training:

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --task-config configs/train/tron2_tasks/my_task.yaml \
  --exp my_task \
  --data-dir "$HF_LEROBOT_HOME" \
  --max-frames 100000
```

Real task YAML files are ignored by `.gitignore`; keep only
`configs/train/tron2_tasks/example.yaml` in the public repository. The template
supports `repo_id`, prompt, dataset column keys, `action_horizon`, `state_dim`,
base checkpoint weights, output directories, `prompt_from_task`, and optional
`rtc_training_simulated_delay`.

## Debug Outputs

When enabled in the YAML, generated runtime files are written under
`debug_images/`:

- `cam_high.jpg`
- `cam_left_wrist.jpg`
- `cam_right_wrist.jpg`
- `tron2_action_data.csv` when `client.save_record: true`
- `tron2_state_data.csv` when `client.save_record: true`
- `tron2_rtc_action_data.csv` and `tron2_rtc_state_data.csv` from RTC runs

These files are local diagnostics and should not be committed.

## Network Deployment Boundary

All current runtime network interfaces—the policy server and client, TRON2
robot control, and Bridge observations—are supported only on a controlled robot
LAN that is accessible to authorized systems. Do not expose these interfaces to
the Internet or use them on an untrusted or shared network.

Not every current transport provides application authentication or TLS. The
policy-serving and robot-control paths must not be treated as authenticated or
encrypted merely because they run inside a LAN. A `wss://` Bridge endpoint does
not secure the other links or expand the supported trust boundary. Any
Internet-facing, cross-site, or cloud topology requires a separate security
review before use.

Source disclosure of the current tasks, prompts, RTC/training design, robot
mapping, calibration, initialization poses, and replay behavior is not a
functional safety approval or real-robot certification. It does not assert that
this repository implements authentication, TLS, emergency stop, motion limits,
collision protection, or watchdog behavior.

See `SECURITY.md` for private vulnerability reporting and the complete
deployment boundary.

## Safety Notes

- Run real-robot clients only with a trained operator present.
- Verify `robot.init_joints` and `robot.init_head` on the target robot before
  executing policy actions.
- Keep the robot workspace clear and keep emergency stop access available.
- Start with short `client.max_steps` values before unlimited execution.
- Use debug images to confirm camera ordering before running a policy.
- This repository does not include private low-level safety controllers.

## Troubleshooting

- If `uv sync` or `uv run` is slow, check package index and network access.
- If the policy cannot load, verify `policy.config`, `policy.repo_id`, and
  `policy.checkpoint_dir`.
- If normalization stats are missing, check
  `checkpoint_dir/assets/<policy.repo_id>/norm_stats.json`.
- If the client cannot connect to the policy server, verify
  `client.policy_host` and `client.policy_port`.
- If bridge observations time out, verify `bridge.host`, TLS settings, and
  Bridge availability.
- If legacy mode misses cameras, verify RealSense serial numbers in
  `camera.serial_to_name`.
- If the robot does not move, verify `robot.ip`, `robot.port`, controller state,
  and that the workspace is safe before retrying.

## Third-Party Origins

This repository is derived from OpenPI and retains upstream OpenPI components.
Some files also include code adapted from Big Vision, HuggingFace Transformers,
LeRobot RTC, Physical Intelligence Kinetix, and `msgpack-numpy`. OpenPI commit
`e01d2290dfef823304b9a59a94b29e5945e38b2d` is the approved working baseline;
the exact origin of each component is not claimed. See `NOTICE`,
`THIRD_PARTY_NOTICES.md`, and `MODIFICATIONS.md` for paths, sources, licenses,
and modification status.

## Contributing

Contributions should stay within the public deployment scope above. Do not
submit private robot profiles, real credentials, internal URLs, customer data,
datasets, model weights, or logs. See `CONTRIBUTING.md` for the base
contribution guidelines.

## License

Source code follows its file-level source licenses. Unless otherwise noted,
project source is provided under the Apache License 2.0 in `LICENSE`;
third-party source remains under the licenses recorded in
`THIRD_PARTY_NOTICES.md` and `LICENSES/`.

`LICENSE_GEMMA.txt` is retained byte-for-byte as upstream-carried model asset
terms material. This source snapshot includes no Gemma or PaliGemma weights,
checkpoints, or model derivatives. External model assets and derivatives use
the applicable Gemma Terms. Those terms do not relicense or add restrictions
to Apache source code. Publishing a model asset, model derivative, or Hosted
Service requires re-review.

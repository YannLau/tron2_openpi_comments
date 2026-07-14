# TRON2 Client Example

This directory contains TRON2 real-robot client examples for the OpenPI-derived
deployment repository. The `tron2_env` runtime is expected to live as a sibling
directory next to `tron2_openpi`:

```text
tron2-vla-open/
├── tron2_openpi/
└── tron2_env/
```

Use the repository-level deployment profile:

```bash
cp config/tron2_deploy.example.yaml config/tron2_deploy.local.yaml
```

Edit `config/tron2_deploy.local.yaml` with your checkpoint, robot address,
observation mode, bridge URL, and camera serials. Do not commit local deployment
profiles.

Start the policy server from the `tron2_openpi` root:

```bash
uv run scripts/serve_policy.py \
  --deploy-config=config/tron2_deploy.local.yaml
```

Start the TRON2 client from another terminal:

```bash
uv run python examples/tron2/pi_client.py \
  --deploy-config=config/tron2_deploy.local.yaml
```

Set `client.observation_source` to `bridge` for TRON2 Bridge WebSocket
observations, or `legacy` for directly attached RealSense cameras.

See the repository `README.md` for the full deployment walkthrough.

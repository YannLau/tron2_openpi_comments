# Contributing to TRON2 OpenPI Deployment

Contributions are welcome when they stay within the public scope of this
repository: TRON2 deployment examples, public configuration templates,
documentation, client-side integration, policy serving, and general OpenPI
compatibility fixes.

## Public Contribution Boundary

Please do not submit:

- Private robot profiles or `.local.yaml` files.
- Real robot IP addresses, camera serial numbers, credentials, or internal URLs.
- Customer data, private logs, debug image captures, datasets, or model weights.
- Undeveloped low-level SDK integrations.
- Safety-critical private controller code or unpublished robot safety policy.

If a change may expose private assets or unclear third-party code, open an issue
or discuss it with the maintainers before submitting a pull request.

## Security Reports

Report any suspected vulnerability privately to
`opensource@limxdynamics.com`. Do not create a public issue for a suspected
vulnerability, and do not include confidential or unredacted material. See
`SECURITY.md` for the reporting and deployment boundaries.

## Issues and Feature Requests

When reporting a bug, include:

- Operating system and Python version.
- The command you ran.
- The relevant public config template fields, with private values redacted.
- The traceback or error message.
- Whether you used bridge observations or legacy RealSense observations.

When requesting a feature, describe:

- The deployment or integration use case.
- The expected behavior.
- Whether the change affects real-robot execution, safety boundaries, or
  third-party dependencies.

## Pull Requests

Before submitting a pull request:

- Keep the change focused.
- Update README or config comments when behavior changes.
- Do not commit generated files such as `.venv`, `__pycache__`, `.DS_Store`,
  debug images, CSV logs, datasets, or checkpoints.
- Run the relevant syntax checks, lint checks, or tests available in your
  environment.
- Include any third-party origin or license notes when adding adapted code.

Unless explicitly stated otherwise, submitted contributions are licensed under
the same license terms as this repository.

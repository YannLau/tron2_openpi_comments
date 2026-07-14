# Modifications from OpenPI

## Comparison basis

- Upstream project: https://github.com/Physical-Intelligence/openpi
- Approved comparison point:
  `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Provenance precision: working baseline + exact origin unknown
- Publication scope: the current file-tree snapshot prepared as a fresh public
  repository; existing Git history and other refs are not included

The commit above is a working baseline for comparison. It is not asserted to be
the exact origin of every inherited file or third-party component.

## Initial comparison evidence

The approved initial comparison recorded:

- 121 common blobs matched the working baseline.
- 16 common paths modified relative to that baseline.
- 18 local additions at the time of the initial comparison.

These are initial comparison counts, not totals for the current snapshot.
Later governance, security, external-dependency, License, and provenance work
added or changed further paths.

## Materially modified common paths in the initial comparison

- `.dockerignore`
- `.gitignore`
- `CONTRIBUTING.md`
- `README.md`
- `examples/droid/main.py`
- `packages/openpi-client/pyproject.toml`
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `pyproject.toml`
- `scripts/serve_policy.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/policies/policy.py`
- `src/openpi/serving/websocket_policy_server.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `uv.lock`

This list records material paths from the approved initial comparison. It does
not imply that unlisted inherited files have an exact known origin.

## LimX derivative areas

The derivative adds or changes TRON2 policy transforms, training and deployment
configuration, robot client examples, Bridge observation integration, RTC
integration, and deployment documentation. Component-level third-party
provenance is recorded in `THIRD_PARTY_NOTICES.md`.

## Later governance and release-readiness changes

Later governance rounds added or changed repository-level materials including:

- `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`, and
  `README_CN.md`
- `.github/ISSUE_TEMPLATE/`, `.github/pull_request_template.md`, and
  `.github/workflows/ci.yml`
- `docs/remote_inference.md` and the two public TRON2 deployment templates
- `examples/aloha_real/README.md` and `examples/libero/README.md` for external
  pinned source checkouts
- `.gitignore`, `NOTICE`, and removal of the stale `.gitmodules` declaration
- `tests/test_repository_readiness.py` and the pytest discovery entry in
  `pyproject.toml`
- `THIRD_PARTY_NOTICES.md`, `MODIFICATIONS.md`, and exact upstream license
  copies under `LICENSES/`
- Final snapshot cleanup in `examples/aloha_real/robot_utils.py` removed a
  non-functional personal-path comment with no runtime behavior change

This section distinguishes later governance files from the initial comparison
counts. It does not claim an exact component origin where the approved evidence
only establishes a working baseline.

# Third-Party Notices

## Scope

This inventory covers only the current source snapshot. It does not cover
wheel/sdist packages, Docker/OCI images, binaries, SDK or firmware, model
weights or checkpoints, model derivatives, datasets, or a Hosted Service.
Third-party code and assets remain under their own license notices; the project
`LICENSE` does not relicense them.

## OpenPI derivative baseline

- Source: https://github.com/Physical-Intelligence/openpi
- Working baseline: `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Origin precision: working baseline + exact origin unknown
- Local path: repository-wide source inherited from or compared with OpenPI;
  material changed paths are listed in `MODIFICATIONS.md`
- Classification: modified vendored / derivative source
- Runtime use: integrated source
- Distributed in: current source snapshot
- Modified: yes
- License: Apache License 2.0 unless an individual file says otherwise
- License file: `LICENSE`
- Notes: the working baseline is a comparison point, not a claim that it is the
  exact origin of every file or copied component

## Big Vision

- Source family: https://github.com/google-deepmind/big_vision
- Reference: inherited from OpenPI working baseline
  `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Origin precision: exact component origin unknown
- Local path: `src/openpi/models/gemma.py`,
  `src/openpi/models/gemma_fast.py`, and `src/openpi/models/siglip.py`
- Classification: adapted source inherited from the OpenPI derivative
- Runtime use: integrated source
- Distributed in: current source snapshot
- Modified: unknown relative to the exact component origin
- Copyright: 2024 Big Vision Authors
- License: Apache License 2.0
- License file: `LICENSE`; individual files retain Apache-2.0 headers

## Google Vision Transformer

- Source: https://github.com/google-research/vision_transformer
- Reference: source path is recorded in the local file; exact component commit
  is unknown
- Origin precision: exact component origin unknown; inherited from OpenPI
  working baseline `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Local path: `src/openpi/models/vit.py`
- Classification: adapted source
- Runtime use: integrated source
- Distributed in: current source snapshot
- Modified: unknown relative to the exact component origin
- Copyright: 2024 Google LLC
- License: Apache License 2.0
- License file: `LICENSE`; the file retains its Apache-2.0 header

## Hugging Face Transformers replacement files

- Source family: https://github.com/huggingface/transformers
- Reference: inherited from OpenPI working baseline
  `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Origin precision: exact component origin unknown
- Local path: `src/openpi/models_pytorch/transformers_replace/`
- Classification: copied / replacement source
- Runtime use: integrated source
- Distributed in: current source snapshot
- Modified: unknown relative to the exact component origin
- Copyright: retained in individual Google and Hugging Face file headers
- License: Apache License 2.0
- License file: `LICENSE`; individual files retain Apache-2.0 headers

## msgpack-numpy

- Source: https://github.com/lebedov/msgpack-numpy
- Reference upstream HEAD prefix: `20c5e5b`
- Origin precision: exact component origin unknown; inherited from OpenPI
  working baseline `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Local path: `packages/openpi-client/src/openpi_client/msgpack_numpy.py`
- Classification: copied / adapted source
- Runtime use: integrated into `openpi-client`
- Distributed in: current source snapshot
- Modified: identical to the OpenPI working baseline; exact component-origin
  diff is unknown
- Copyright: 2013-2022 Lev E. Givon
- License: BSD-3-Clause
- License file: `LICENSES/msgpack-numpy-BSD-3-Clause.txt`

## ACT / ALOHA copied files

- Source: https://github.com/tonyzhaozh/act
- Reference upstream commit prefix: `742c753`
- Origin precision: exact component origin unknown; inherited from OpenPI
  working baseline `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Local path: `examples/aloha_real/constants.py`,
  `examples/aloha_real/real_env.py`, and
  `examples/aloha_real/robot_utils.py`
- Classification: copied source
- Runtime use: ALOHA real-robot example
- Distributed in: current source snapshot
- Modified: `examples/aloha_real/constants.py` and
  `examples/aloha_real/real_env.py` remain identical to the OpenPI working
  baseline. `examples/aloha_real/robot_utils.py` is locally modified only by
  removal of a non-functional personal-path comment, with no runtime behavior
  change. The exact component-origin diff remains unknown.
- Copyright: 2023 Tony Z. Zhao
- License: MIT
- License file: `LICENSES/ACT-MIT.txt`

## DROID copied section

- Source:
  https://github.com/JonathanYang0127/r2d2_rlds_dataset_builder/blob/parallel_convert/r2_d2/r2_d2.py
- Reference upstream commit prefix: `e9254e3`
- Origin precision: exact component origin unknown; inherited from OpenPI
  working baseline `e01d2290dfef823304b9a59a94b29e5945e38b2d`
- Local path: parsing section of
  `examples/droid/convert_droid_data_to_lerobot.py`
- Classification: copied source section
- Runtime use: DROID dataset conversion example
- Distributed in: current source snapshot
- Modified: identical to the OpenPI working baseline; exact component-origin
  diff is unknown
- Copyright: 2023 Karl Pertsch
- License: MIT
- License file: `LICENSES/DROID-MIT.txt`

## robosuite snippet

- Source:
  https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py
- Pinned source commit: `eafb81f54ffc104f905ee48a16bb15f059176ad3`
- Local path: `_quat2axisangle` in `examples/libero/main.py`
- Classification: copied source snippet
- Runtime use: LIBERO example
- Distributed in: current source snapshot
- Modified: inherited from the OpenPI working baseline; no separate local
  component-origin diff is claimed
- Copyright: 2022 Stanford Vision and Learning Lab and UT Robot Perception and
  Learning Lab
- License: MIT; the upstream license also carries a DeepMind MuJoCo Apache-2.0
  notice
- License file: `LICENSES/robosuite-MIT-and-Apache-2.0.txt`

## Kinetix RTC

- Source: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
- Reference upstream commit prefix: `9296f31`
- Origin precision: exact component origin unknown
- Local path: `src/openpi/rtc/rtc_processor.py`
- Classification: locally added adapted source
- Runtime use: integrated real-time chunking
- Distributed in: current source snapshot
- Modified: yes; local addition based on the referenced implementation
- Copyright: 2025 Physical Intelligence
- License: MIT
- License file: `LICENSES/Kinetix-MIT.txt`

## LeRobot RTC

- Source: https://github.com/huggingface/lerobot
- Reference upstream commit prefix: `e40b58a`
- Origin precision: exact component origin unknown
- Local path: `src/openpi/rtc/rtc_config.py` and
  `src/openpi/rtc/rtc_processor.py`
- Classification: locally added adapted source
- Runtime use: integrated real-time chunking
- Distributed in: current source snapshot
- Modified: yes; local additions based on the referenced implementation
- Copyright: 2024 The Hugging Face team
- License: Apache License 2.0
- License file: `LICENSE`; local files identify the LeRobot source paths

## External ALOHA checkout

- Source: https://github.com/Physical-Intelligence/aloha.git
- Pinned commit: `d1dc83afd89ded4379851257fe5d85632d31d5ec`
- Local path when installed by the user: `third_party/aloha`
- Classification: external dependency / external source checkout
- Runtime use: installed by user for the ALOHA example
- Distributed in: not included in the current source snapshot
- Modified: not applicable to this snapshot
- License: MIT in the external repository
- License file: provided by the external checkout, not copied into this snapshot

## External LIBERO checkout

- Source: https://github.com/Lifelong-Robot-Learning/LIBERO.git
- Pinned commit: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Local path when installed by the user: `third_party/libero`
- Classification: external dependency / external source checkout
- Runtime use: installed by user for the LIBERO example
- Distributed in: not included in the current source snapshot
- Modified: not applicable to this snapshot
- License: MIT for the external code repository; no LIBERO dataset is included
- License file: provided by the external checkout, not copied into this snapshot

## Gemma model asset terms material

- Local path: `LICENSE_GEMMA.txt`
- Classification: upstream-carried model asset terms material, not the source
  code license for Apache-licensed files
- Distributed in: current source snapshot as terms reference material
- Model assets distributed in: none; no Gemma or PaliGemma weights,
  checkpoints, or model derivatives are included
- Source-code license: each source file remains governed by its file-level
  source license
- Terms boundary: external Gemma/PaliGemma model assets and derivatives use the
  applicable Gemma Terms; those terms do not relicense or add restrictions to
  Apache source code
- Future scope: publishing a model asset, model derivative, or Hosted Service
  requires re-review

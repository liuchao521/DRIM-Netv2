# Data preparation

This repository expects pre-extracted face images and does not redistribute benchmark data.

1. Obtain each dataset from its official provider and follow its license.
2. Extract and align face crops using one fixed preprocessing pipeline for all compared methods.
3. Create train, validation, and test list files using the format shown in `splits/example.txt`.
4. Use frame-disjoint or video-disjoint splits required by the benchmark protocol.

Metrics in the paper are reported under the corresponding dataset protocols. Do not select thresholds or hyperparameters on target-domain test labels.


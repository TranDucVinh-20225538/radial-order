# Round 3 report

Runtime: 1719.3 s

## Part 1 — extra disjoint metrics

- **dino_derm / diag**: median Gamma [0.619, 0.633] median=0.627; vs 1/2: above_1/2; vs own null: above_own_null; max identity_gap=0.0000
- **dino_derm / sphere_LW**: median Gamma [0.152, 0.167] median=0.163; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000
- **dino_path / diag**: median Gamma [0.562, 0.582] median=0.569; vs 1/2: above_1/2; vs own null: above_own_null; max identity_gap=0.0000
- **dino_path / sphere_LW**: median Gamma [0.355, 0.385] median=0.372; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000
- **clip_derm / diag**: median Gamma [0.474, 0.500] median=0.490; vs 1/2: mixed; vs own null: mixed_vs_null; max identity_gap=0.0000
- **clip_derm / sphere_LW**: median Gamma [0.034, 0.046] median=0.043; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000
- **clip_path / diag**: median Gamma [0.656, 0.666] median=0.660; vs 1/2: above_1/2; vs own null: above_own_null; max identity_gap=0.0000
- **clip_path / sphere_LW**: median Gamma [0.204, 0.229] median=0.215; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000
- **dino_iwild / diag**: median Gamma [0.417, 0.440] median=0.431; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000
- **dino_iwild / sphere_LW**: median Gamma [0.329, 0.358] median=0.346; vs 1/2: below_1/2; vs own null: below_own_null; max identity_gap=0.0000

Output: `extra_metrics_disjoint.csv`.

## Part 2 — surrogate bands (seed 0 disjoint LW)

- **dino_derm**: q=0.1 G_low mean=0.100 (|Gamma-full|=0.100 vs |real-full|=0.152); q=0.1 G_high mean=0.002 (|Gamma-full|=0.002 vs |real-full|=0.152); q=0.1 P_low mean=0.100 (|Gamma-full|=0.100 vs |real-full|=0.152); q=0.25 G_low mean=0.038 (|Gamma-full|=0.038 vs |real-full|=0.152); q=0.25 G_high mean=0.027 (|Gamma-full|=0.027 vs |real-full|=0.152); q=0.25 P_low mean=0.037 (|Gamma-full|=0.037 vs |real-full|=0.152); q=0.5 G_low mean=0.002 (|Gamma-full|=0.002 vs |real-full|=0.152); q=0.5 G_high mean=0.089 (|Gamma-full|=0.089 vs |real-full|=0.152); q=0.5 P_low mean=0.001 (|Gamma-full|=0.001 vs |real-full|=0.152)
- **dino_path**: q=0.1 G_low mean=0.365 (|Gamma-full|=0.091 vs |real-full|=0.097); q=0.1 G_high mean=0.268 (|Gamma-full|=0.005 vs |real-full|=0.097); q=0.1 P_low mean=0.369 (|Gamma-full|=0.095 vs |real-full|=0.097); q=0.25 G_low mean=0.353 (|Gamma-full|=0.079 vs |real-full|=0.097); q=0.25 G_high mean=0.290 (|Gamma-full|=0.016 vs |real-full|=0.097); q=0.25 P_low mean=0.361 (|Gamma-full|=0.088 vs |real-full|=0.097); q=0.5 G_low mean=0.319 (|Gamma-full|=0.045 vs |real-full|=0.097); q=0.5 G_high mean=0.326 (|Gamma-full|=0.052 vs |real-full|=0.097); q=0.5 P_low mean=0.330 (|Gamma-full|=0.056 vs |real-full|=0.097)
- **clip_derm**: q=0.1 G_low mean=0.031 (|Gamma-full|=0.031 vs |real-full|=0.045); q=0.1 G_high mean=0.000 (|Gamma-full|=0.000 vs |real-full|=0.045); q=0.1 P_low mean=0.031 (|Gamma-full|=0.031 vs |real-full|=0.045); q=0.25 G_low mean=0.011 (|Gamma-full|=0.011 vs |real-full|=0.045); q=0.25 G_high mean=0.001 (|Gamma-full|=0.001 vs |real-full|=0.045); q=0.25 P_low mean=0.010 (|Gamma-full|=0.010 vs |real-full|=0.045); q=0.5 G_low mean=0.001 (|Gamma-full|=0.001 vs |real-full|=0.045); q=0.5 G_high mean=0.009 (|Gamma-full|=0.009 vs |real-full|=0.045); q=0.5 P_low mean=0.000 (|Gamma-full|=0.000 vs |real-full|=0.045)
- **clip_path**: q=0.1 G_low mean=0.219 (|Gamma-full|=0.200 vs |real-full|=0.213); q=0.1 G_high mean=0.027 (|Gamma-full|=0.008 vs |real-full|=0.213); q=0.1 P_low mean=0.219 (|Gamma-full|=0.200 vs |real-full|=0.213); q=0.25 G_low mean=0.182 (|Gamma-full|=0.163 vs |real-full|=0.213); q=0.25 G_high mean=0.075 (|Gamma-full|=0.056 vs |real-full|=0.213); q=0.25 P_low mean=0.182 (|Gamma-full|=0.163 vs |real-full|=0.213); q=0.5 G_low mean=0.103 (|Gamma-full|=0.084 vs |real-full|=0.213); q=0.5 G_high mean=0.154 (|Gamma-full|=0.135 vs |real-full|=0.213); q=0.5 P_low mean=0.097 (|Gamma-full|=0.078 vs |real-full|=0.213)
- **dino_iwild**: q=0.1 G_low mean=0.320 (|Gamma-full|=0.228 vs |real-full|=0.255); q=0.1 G_high mean=0.146 (|Gamma-full|=0.055 vs |real-full|=0.255); q=0.1 P_low mean=0.320 (|Gamma-full|=0.228 vs |real-full|=0.255); q=0.25 G_low mean=0.261 (|Gamma-full|=0.169 vs |real-full|=0.255); q=0.25 G_high mean=0.255 (|Gamma-full|=0.163 vs |real-full|=0.255); q=0.25 P_low mean=0.258 (|Gamma-full|=0.166 vs |real-full|=0.255); q=0.5 G_low mean=0.137 (|Gamma-full|=0.046 vs |real-full|=0.255); q=0.5 G_high mean=0.315 (|Gamma-full|=0.224 vs |real-full|=0.255); q=0.5 P_low mean=0.102 (|Gamma-full|=0.011 vs |real-full|=0.255)

Outputs: `surrogate_bands.csv`, `surrogate_hybrid.csv`.

## Part 3 — CLIP ViT-B/16 iWildCam

Recovered subsample: `iwildcam/indices_id.npy` and `indices_new.npy` (sha256 f64842c1406801822490f98b1891fc36fe207db4bf2135ca227227f8bf670e27, 1f37584d022ce50931b2b055790c51792c5c37c9a3ab482e2e2bb4fedf1a3fd6); train vs official test per `IWILDCAM.md` (cap: sort, RandomState(0).shuffle, first 2000).
New embeddings: `export_embeddings/clip_iwild_id.npy` sha256=a5f287d10799e2a47047443f5f604573f07a63de23b5c16c73638fbac7debe6a; `clip_iwild_new.npy` sha256=49a063027338cccf9a5c61a75c4e744b1d4ec9cb0993b9c2540725dfb01ee1a6.
Outputs: `clip_iwild.csv`, `clip_iwild_path.csv`, `clip_iwild_perm.csv`.

## New .npy sha256

- export_embeddings/clip_iwild_id.npy: `a5f287d10799e2a47047443f5f604573f07a63de23b5c16c73638fbac7debe6a`
- export_embeddings/clip_iwild_new.npy: `49a063027338cccf9a5c61a75c4e744b1d4ec9cb0993b9c2540725dfb01ee1a6`

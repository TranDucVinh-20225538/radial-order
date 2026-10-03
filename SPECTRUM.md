# Spectrum

Stopped. The raw embedding matrices that were scored were not saved.

`embed_paths` returns the CLS float32 clouds in memory. `row_for` writes only the ID Ledoit–Wolf matrix, `artifacts/sigma_{modality}_{split}.npy`. No `X_ID` or `X_new` file was written for dermatology or for pathology.

| modality | status |
| --- | --- |
| dermatology | matrices_absent |
| pathology | matrices_absent |

Both modalities are absent, so no second-moment matrix was formed and `scipy.linalg.eigh` was not called. Ledoit–Wolf was not refit. The saved `sigma_*.npy` files are the ID covariances, not the clouds, and were not used as a substitute.

SciPy version installed in this environment: 1.17.1. Divisor would have been `n_new`. It was not applied.

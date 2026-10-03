# Results

Euclidean AUROC is reported and is not part of pass. The oracle map is an alignment, not a detector.

| modality | split | n_id | n_new | premap_mw | certificate | epsilon | auroc_mahalanobis | auroc_gaussian | gap | auroc_euclidean | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dermatology | smoke | 64 | 64 | 0 | no | none | 0 | 0 | 0 | 0.632568359375 | yes |
| dermatology | isic2019_padufes | 2000 | 2000 | 0.01547975 | no | none | 0.01547975 | 0.01547975 | 0 | 0.661086 | yes |
| dermatology | control | 2000 | 2000 | 0 | no | none | 0 | 0 | 0 | 0 | yes |
| pathology | camelyon17_train_test | 2000 | 2000 | 0.22347525 | no | none | 0.22347525 | 0.22347525 | 0 | 0.58283075 | yes |

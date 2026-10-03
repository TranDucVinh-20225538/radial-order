# Re-embed

hostname: node002
torch: 2.8.0+cu128
scipy: 1.17.1
filename_list_source: saved list
cap_redraw: no
encoder: torch.hub facebookresearch/dinov2 dinov2_vitb14
eigh: ran

Scored AUROC values are the finished-run figures and are not overwritten.

## dermatology

list_id: /data2/cmdir/home/toandq/radial-order/filename_lists/dermatology_id.txt
list_new: /data2/cmdir/home/toandq/radial-order/filename_lists/dermatology_new.txt
sha256_X_id: 6805481fd5f640f895a04bf767116a32ea107f0c7047a198598279167a60122e
sha256_X_new: 41279576989ee5527f6316230528162068a9d2c521905158f7a95a0e3d50d379
sha256_filenames_id: dc9f6fe62865b05fc705794e73ff2e2a02859c8abf7ec04984d58a5c408c50ba
sha256_filenames_new: db49b1287330b9bb08ca9ddf40d020ca9a3e404ac505dadea31f6cdbb82076b2
n_id: 2000
n_new: 2000
d: 768
sign_condition: straddles
lambda_min: 2.523673348061306e-12
lambda_max: 104.6467441496339
trace_over_d: 3.030901204730673
auroc_mahalanobis: 0.01547974999999999
premap_mw: 0.01547975
scored_auroc: 0.0155

## pathology

list_id: /data2/cmdir/home/toandq/radial-order/filename_lists/pathology_id.txt
list_new: /data2/cmdir/home/toandq/radial-order/filename_lists/pathology_new.txt
sha256_X_id: 57e206e956f9ccc6bdc162ddf7bf0be0ea98e923447d4ee5844c700ec66770e8
sha256_X_new: 26ae000727ecb604e51abec9e1032ae6f5ddeeea7711539882ec5c6ae71921fc
sha256_filenames_id: de763c655d2d4b5947676109ee32657aa7d408a8d1eeb53560e07ffc15aa2c30
sha256_filenames_new: f69cce2e1740e1fd3e3148ab58c6585a9deae88d4aae9104b19fa78e28cfe102
n_id: 2000
n_new: 2000
d: 768
sign_condition: straddles
lambda_min: 5.506887914418789e-12
lambda_max: 65.64998740180977
trace_over_d: 1.8138067192856
auroc_mahalanobis: 0.22347525
premap_mw: 0.22347525
scored_auroc: 0.2235

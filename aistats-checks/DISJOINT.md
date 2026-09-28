# Disjoint-fit radii

CPU only. Saved matrices only. No ridge. No re-embed.

Real protocol: RandomState(0) permutation of X_id; A=first half, B=second half.
Fit (μ, Σ_LW) on A. μ_new = column mean of full X_new. ID radii on B; new radii on X_new.
Euclidean uses Σ=I with the same centers.

Matched null: same RandomState(0) permutation of X_id;
A=first 1000 (fit), B=next 500 (ID eval), C=last 500 (pseudo-new). All disjoint.
This matches the real-cell fit size (1000). Eval sizes are smaller than the real new cloud.

The earlier half/half null in null_split.csv fits and scores on overlapping roles;
it is not overwritten. These files are the comparable pair.

## dino_derm

published_lw: 0.0155
disjoint_real_lw: 0.15196199999999999
matched_null_lw: 0.5137200000000001
n_fit: 1000
d_over_n_fit: 0.768
relation_to_this_cell_null: below
flags: none

## dino_path

published_lw: 0.2235
disjoint_real_lw: 0.371211
matched_null_lw: 0.5048360000000001
n_fit: 1000
d_over_n_fit: 0.768
relation_to_this_cell_null: below
flags: none

## clip_derm

published_lw: 0.00504
disjoint_real_lw: 0.044637
matched_null_lw: 0.5029720000000001
n_fit: 1000
d_over_n_fit: 0.512
relation_to_this_cell_null: below
flags: none

## clip_path

published_lw: 0.1108
disjoint_real_lw: 0.23225
matched_null_lw: 0.5175599999999999
n_fit: 1000
d_over_n_fit: 0.512
relation_to_this_cell_null: below
flags: none

## dino_iwild

published_lw: 0.1427
disjoint_real_lw: 0.3464
matched_null_lw: 0.5164359999999999
n_fit: 1000
d_over_n_fit: 0.768
relation_to_this_cell_null: below
flags: none

## Summary

matched_null_lw_min: 0.5029720000000001
matched_null_lw_max: 0.5175599999999999
disjoint_real_lw_min: 0.044637
disjoint_real_lw_max: 0.371211

No ranking of metrics. Theorem 1 is untouched.
Published LW numbers are left as published.

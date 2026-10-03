# iWildCam

subsets: train, test
id: get_subset("train")
new: get_subset("test")
cap: sort indices, RandomState(0).shuffle, first 2000
sha256_indices_id: f64842c1406801822490f98b1891fc36fe207db4bf2135ca227227f8bf670e27
sha256_indices_new: 1f37584d022ce50931b2b055790c51792c5c37c9a3ab482e2e2bb4fedf1a3fd6
sha256_X_id: 91af85edf19d0a8bf97473cbef09b491e09533ad3eac03a2a903d8d947bffa23
sha256_X_new: 0387232b1b5c758e316f16d033c75241315641e6d449cc90efb386b982ed2a4d
torch: 2.8.0+cu128
scipy: 1.17.1
hostname: node002
encoder: torch.hub facebookresearch/dinov2 dinov2_vitb14
vector: x_norm_clstoken, float32, no later L2
eigh: ran
n_id: 2000
n_new: 2000
d: 768
ledoit_gamma: 0.14271625
ledoit_gaussian_gap: 0
ledoit_same_sign: yes
euclidean_gamma: 0.4314165
euclidean_gaussian_gap: 0
euclidean_same_sign: yes
sign_condition: straddles
n_indices_id: 2000
n_indices_new: 2000

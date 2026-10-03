# CLIP

checkpoint: ViT-B/16
vector: model.encode_image, float32, no later L2
preprocess: Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    <function _convert_image_to_rgb at 0x74c0ba82f2e0>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
) | Resize size=224 | CenterCrop size=(224, 224) | function | ToTensor | Normalize mean=(0.48145466, 0.4578275, 0.40821073) std=(0.26862954, 0.26130258, 0.27577711)
filename_list_source: job 61185 saved list
torch: 2.8.0+cu128
scipy: 1.17.1
hostname: node002
eigh: ran

## dermatology

list_id: /data2/cmdir/home/toandq/radial-order/reembed/dermatology/filenames_id.txt
list_new: /data2/cmdir/home/toandq/radial-order/reembed/dermatology/filenames_new.txt
sha256_X_id: 085bd514c53098baa94abf32d51c7b9cb58e99247ff72977be944231585f9ff2
sha256_X_new: 546ad3750f9ac9589174458a36c8b71b8409fb3a7b28077a0f4b6d309b2b00b4
sha256_filenames_id: dc9f6fe62865b05fc705794e73ff2e2a02859c8abf7ec04984d58a5c408c50ba
sha256_filenames_new: db49b1287330b9bb08ca9ddf40d020ca9a3e404ac505dadea31f6cdbb82076b2
n_id: 2000
n_new: 2000
d: 512
ledoit_gamma: 0.005041499999999994
ledoit_gaussian_gap: 0
ledoit_same_sign: yes
euclidean_gamma: 0.50140575
euclidean_gaussian_gap: 0
euclidean_same_sign: yes
sign_condition: straddles

## pathology

list_id: /data2/cmdir/home/toandq/radial-order/reembed/pathology/filenames_id.txt
list_new: /data2/cmdir/home/toandq/radial-order/reembed/pathology/filenames_new.txt
sha256_X_id: 9e337cd74c0acbb99d93e7df5edf998027e6023b71d7ac03b93f7740dcd1f5cf
sha256_X_new: 14ad3f676fb10a6ba6a3d8eb775f5edad38978c46891d5a640445a6e85bafbf8
sha256_filenames_id: de763c655d2d4b5947676109ee32657aa7d408a8d1eeb53560e07ffc15aa2c30
sha256_filenames_new: f69cce2e1740e1fd3e3148ab58c6585a9deae88d4aae9104b19fa78e28cfe102
n_id: 2000
n_new: 2000
d: 512
ledoit_gamma: 0.11083775
ledoit_gaussian_gap: 0
ledoit_same_sign: yes
euclidean_gamma: 0.7568189999999999
euclidean_gaussian_gap: 0
euclidean_same_sign: yes
sign_condition: straddles

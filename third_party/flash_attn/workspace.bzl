"""Provides the repository macro to import flash-attention."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Path to the PyTorch project relative to the XLA project's `WORKSPACE` file.
PYTORCH_LOCAL_DIR = "../pytorch/"

def repo():
    # Setup PyTorch if there is no PyTorch repository already defined(from torch_xla etc.)
    maybe(
        repo_rule = native.new_local_repository,
        name = "torch",
        build_file = "//third_party/flash_attn:torch.BUILD",
        path = PYTORCH_LOCAL_DIR,
    )

    # v2.5.9.post1
    FLASH_ATTN_COMMIT = "e2e4333c955b829d0e6087d27ee435f55c80d3a5"
    FLASH_ATTN_SHA256 = "0611ec789aa8b837d88209a51c172fecc1a91b54b6bf5a21070c5867005eb079"

    tf_http_archive(
        name = "flash_attn",
        sha256 = FLASH_ATTN_SHA256,
        strip_prefix = "flash-attention-{commit}".format(commit = FLASH_ATTN_COMMIT),
        urls = tf_mirror_urls("https://github.com/Dao-AILab/flash-attention/archive/{commit}.tar.gz".format(commit = FLASH_ATTN_COMMIT)),
        build_file = "//third_party/flash_attn:flash_attn.BUILD",
        patch_file = [
            "//third_party/flash_attn:flash_attn.patch"
        ],
    )


for feature_emb_name, embedding_table in emb_tables.items () :
    if feature_emb_name != "common_feature_hash_key": continue
embedding_list = fuse_safe_embedding_lookup_sparse(embedding_table, fuse_feature_input_dict [feature_emb_name], None, combiner='sum')
f_deep_id_embedding_normal.extend (embedding_list)


from tensorflow.python.ops import sparse_ops
common_feature_input_list = []
common_embedding_table = None
item_feature_input_list = []
item_embedding_table = None
for feature_emb_name, embedding_table in emb_tables.items () :
    if feature_emb_name == "common_feature_hash_key":
        common_feature_input_list.append(fuse_feature_input_dict[feature_emb_name])
        common_embedding_table = embedding_table
    else:
        item_feature_input_list.append(fuse_feature_input_dict[feature_emb_name])
        item_embedding_table = embedding_table

# for common
fuse_common_inputs = sparse_ops.sparse_concat(0, common_feature_input_list, expand_nonconcat_dim=True)
common_fuse_embeddings = common_embedding_table(fuse_common_inputs)
common_embedding_list = array_ops.split(common_fuse_embeddings, num_or_size_splits=num_split, axis=0)

# for item
fuse_item_inputs = sparse_ops.sparse_concat(0, item_feature_input_list, expand_nonconcat_dim=True)
item_fuse_embeddings = item_embedding_table(fuse_item_inputs)
item_embedding_list = array_ops.split(item_fuse_embeddings, num_or_size_splits=num_split, axis=0)
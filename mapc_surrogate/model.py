import jax.numpy as jnp
import jraph
import flax.linen as nn
from jraph._src.utils import segment_mean


class FeedForwardBlock(nn.Module):
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.ff_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    n_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=x.shape[-1],
            dtype=self.dtype,
            use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        x = nn.Dense(self.dim, dtype=self.dtype, use_bias=False)(x)

        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.ff_dim, self.drop_rate, self.dtype)(x, mask=mask, training=training)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


def compute_segment_ids(n_elements, total_length):
    return jnp.searchsorted(jnp.cumsum(n_elements), jnp.arange(total_length), side='right')


def make_transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype, segment_ids=None, training=True):
    if segment_ids is not None:
        mask = segment_ids[:, None] == segment_ids[None, :]
    else:
        mask = None

    @jraph.concatenated_args
    def update_fn(inputs):
        return Transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype)(inputs, mask=mask, training=training)

    return update_fn


def make_embed_fn(latent_size, dtype):
    @jraph.concatenated_args
    def embed_fn(inputs):
        return nn.Dense(latent_size, dtype=dtype)(inputs)

    return embed_fn


class SurrogateModel(nn.Module):
    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    n_components: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        x = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.dim, self.dtype))(x)

        transformer_fn = lambda segment_ids: make_transformer(
            self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, segment_ids, training=training
        )

        edge_segment_ids = compute_segment_ids(x.n_edge, x.senders.shape[0])
        node_segment_ids = compute_segment_ids(x.n_node, x.nodes.shape[0])
        global_segment_ids = jnp.arange(x.n_node.shape[0])

        for _ in range(self.gnn_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=transformer_fn(edge_segment_ids),
                update_node_fn=transformer_fn(node_segment_ids),
                update_global_fn=transformer_fn(global_segment_ids),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)

        x = jraph.GraphMapFeatures(embed_global_fn=make_embed_fn(self.n_components * 3, self.dtype))(x)

        logits, means, raw_scales = jnp.split(x.globals, 3, axis=-1)
        scales = nn.softplus(raw_scales) + 1e-4
        return logits, means, scales

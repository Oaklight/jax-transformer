import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from einops import rearrange, repeat

from typing import Optional, Any, Sequence
import dataclasses

import matplotlib.pyplot as plt

class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: Optional[int]=None,
        value_size: Optional[int]=None,
        model_size: Optional[int]=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size or (model_size // num_heads) if model_size is not None else None
        self.value_size = value_size or key_size
        self.model_size = model_size or (key_size * num_heads) if key_size is not None else None
        assert self.model_size or self.key_size is not None, "must provide at least one of `key_value`, `model_size`"
        assert key_size * num_heads == model_size, "Must satisfy `key_size * num_heads == model_size`"
    
    @hk.transparent
    def __linear(
        self,
        x: jnp.ndarray, # [b, n_q, embed_size] 
        head_size: int,
    ) -> jnp.ndarray: # [b, num_head, n_q, head_size]
        '''
        map a single embedded input sequence into a size `num_heads` collection of featured tensors
        shape change: [b, n, embed_size] -> [b, n, num_heads*head_size] -> [b, num_heads, n, head_size]
        - b: batch_size
        - n: sequence length
        '''
        y = hk.Linear(self.num_heads * head_size)(x) # shape of y: [b, n, num_heads*head_size]
        return rearrange(y, 'b n (h d) -> b h n d', h=self.num_heads) # shape of return: [b, num_heads, n, head_size]
    
    def __call__(
        self,
        q_input: jnp.ndarray, # shape: [b, n_q, embed_size]
        k_input: jnp.ndarray, # shape: [b, n_k, embed_size]
        v_input: jnp.ndarray, # shape: [b, n_k, embed_size]
        mask: Optional[jnp.ndarray] = None, # shape: [b, 1, n_q, n_k]
    ) -> jnp.ndarray:
        '''
        - map embedded input sequences (q_input, k_input, v_input) into multi-headed featured tensors (q, k, v)
        - compute attention weights by "compatible function of queries and keys": w_ij = (q_ik*k_jk) / sqrt(key_size)
        - compute "weighted sum of values": ret_ij = w_ik * v_kj
        - map concatenated "weighted sum of values" to a feature tensor
        '''
        q = self.__linear(q_input, self.key_size) # shape: [b, n_q, embed_size] -> [b, num_head, n_q, key_size]
        k = self.__linear(k_input, self.key_size) # shape: [b, n_k, embed_size] -> [b, num_head, n_k, key_size]
        v = self.__linear(v_input, self.value_size) # shape: [b, n_k, embed_size] -> [b, num_head, n_k, value_size]
        # Q: why n_k not n_v?
        # A: key value pairs are one to one mapping, n_v == n_k
        
        attn_weights = jnp.einsum('bhik, bhjk -> bhij', q, k) # shape: [b, h, n_q, key_size] @ [b, h, n_k, key_size] -> [b, h, n_q, n_k]
        attn_weights /= jnp.sqrt(self.key_size)
        
        if mask is not None:
            assert mask.ndim == attn_weights.ndim, f"Mask dimensionality {mask.ndim} must match attention weights dimensionality {attn_weights.ndim}"
            attn_weights = jnp.where(mask, attn_weights, -1e30)
        attn_weights = jax.nn.softmax(attn_weights) # [b, h, n_q, n_k]
        
        weighted_v = jnp.einsum('bhik, bhkd -> bhid', attn_weights, v) # shape: [b, h, n_q, n_k] @ [b, h, n_k, value_size] -> [b, h, n_q, value_size]
        weighted_v = rearrange(weighted_v, 'b h n v -> b n (h v)') # shape: [b, h, n_q, value_size] -> [b, n_q, h*value_size]
        
        v_ret = hk.Linear(self.model_size)(weighted_v) # shape: [b, n_q, h*value_size] -> [b, n_q, model_size]
        return v_ret
    

    
class FFN(hk.Module):
    def __init__(
        self,
        model_size: int,
        hidden_size: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.wb1 = hk.Linear(hidden_size) # W1 & b1
        self.wb2 = hk.Linear(model_size) # W2 & b2
        self.dropout_rate = dropout_rate
        
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        y = self.wb1(x) # xW1 + b1
        y = jax.nn.relu(y) # max(0, xW1 + b1)
        y = self.wb2(y) # max(0, xW1 + b1)W2 + b2
        y = hk.dropout(hk.next_rng_key(), self.dropout_rate, y)
        return y
    
    

@dataclasses.dataclass
class PositionalEncoding(hk.initializers.Initializer):
    '''
    - has the same model_size as embedding
    - use sine and cosine functions of different frequencies:
        - PE(pos, 2i) = sin(pos/10000^(2i/model_size))
        - PE(pos, 2i+1) = cos(pos/10000^(2i/model_size))
    '''
    def __call__(
        self,
        shape: Sequence[int],
        dtype: Any = np.float32,
    ) -> jnp.ndarray:
        max_len, model_size = shape
        pe = np.zeros((max_len, model_size), dtype=dtype)
        # pos = np.expand_dims(np.arange(0, max_len), 1)
        pos = np.arange(0, max_len)[:,np.newaxis]
        div_term = np.exp(np.arange(0, model_size, 2) * 
                         -(np.log(10000.) / model_size)) # 1/(10000^(2i/model_size)) = 10000^(-2i/model_size) = exp(-2i * ln(10000) / model_size)
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        pe = jnp.array(pe[np.newaxis:])
        pe = jax.lax.stop_gradient(pe) # this is a frozen parameter
        
        return pe
        
        # print(type(x))
        # x = x + self.pe[:, :x.size(1)]
        # return hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        
        
class Embeddings(hk.Module):
    def __init__(
        self,
        model_size: int,
        vocab_size: int,
        embedding_matrix: Optional[jnp.ndarray] = None,
    ):
        '''
        if embedding-tying, then use generator's weights as embedding_matrix
        '''
        super().__init__()
        if embedding_matrix is None:
            self.embed = hk.Embed(
                vocab_size=vocab_size,
                embed_dim=model_size,
            )
        else:
            self.embed = hk.Embed(
                embedding_matrix=embedding_matrix
            )
        self.model_size = model_size

    def __call__(
        self,
        x: jnp.ndarray # shape [b, n]
    ) -> jnp.ndarray: # shape [b, n, embed_size], embed_size == model_size
        '''
        - In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
        - "In the embedding layers, we multiply those weights by sqrt(model_size)"
        '''    
        return self.embed(x) * jnp.sqrt(self.model_size)
    
    

@dataclasses.dataclass
class TransformerConfig:
    '''
    global hyperparameters used to minimize obnoxious kwarg plumbing
    '''
    vocab_size: int
    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    hidden_size: int = 2048
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        '''
        sanity check and compute missing hyperparameters
        '''
        # recall the `rearrange` function in multi-head attention: `b n (hd) -> b h n d`
        self.key_size = self.model_size // self.num_heads
        # this is because the transformer is designed to have same-size key and value, 
        #     not because attention module should have same-size key and value
        self.value_size = self.key_size
    
    def __repr__(self):
        '''
        print out all hyperparameters
        '''
        return str(self.__dict__)

    
    
def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    '''
    Applies a unique LayerNorm to x with default settings.
    '''
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)

@dataclasses.dataclass
class Encoder(hk.Module):
    config: TransformerConfig
    
    def __call__(
        self,
        src_embed: jnp.ndarray, # [b, n, embed_size], embed_size == model_size
        src_mask: Optional[jnp.ndarray] = None, # [b, 1, n, n]
        is_training: bool = True,
    ) -> jnp.ndarray:
        config = self.config
        
        dropout_rate = config.dropout_rate if is_training else 0
        
        h = src_embed
        for _ in range(config.num_layers):
            
            # multi-head attention SUBLAYER
            attn_block = MultiHeadAttention(
                num_heads=config.num_heads,
                key_size=config.key_size,
                model_size=config.model_size,
            )
            # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            h_attn = attn_block(h, h, h, mask=src_mask) # [b, n, model_size]
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = layer_norm(h + h_attn)
            
            # position-wise ffn SUBLAYER
            ffn_block = FFN(
                model_size=config.model_size,
                hidden_size=config.hidden_size,
                dropout_rate=config.dropout_rate,
            )
            # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            h_dense = ffn_block(h) # [b, n, model_size]
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = layer_norm(h + h_dense) # [b, n, model_size]
        
        return h
    


@dataclasses.dataclass
class Decoder(hk.Module):
    config: TransformerConfig
    
    def __call__(
        self,
        tgt_embed: jnp.ndarray, # [b, n_q, embed_size], embed_size == model_size
        src_encoded: jnp.ndarray, # [b, n_k, model_size], encoder output of src_embed
        tgt_mask: Optional[jnp.ndarray] = None, # [b, 1, n_q, n_q]
        tgt_src_mask: Optional[jnp.ndarray] = None, # [b, 1, n_q, n_k]
        is_training: bool = True,
    ) -> jnp.ndarray:
        config = self.config
        
        dropout_rate = config.dropout_rate if is_training else 0
        
        h = tgt_embed
        for _ in range(config.num_layers):
            
            # multi-head attention SUBLAYER
            attn_block = MultiHeadAttention(
                num_heads=config.num_heads,
                key_size=config.key_size,
                model_size=config.model_size,
            )
            # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            h_attn = attn_block(h, h, h, mask=tgt_mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = layer_norm(h + h_attn)
            
            # multi-head cross attention SUBLAYER
            cross_attn_block = MultiHeadAttention(
                num_heads=config.num_heads,
                key_size=config.key_size,
                model_size=config.model_size,
            )
            # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            h_attn = cross_attn_block(h, src_encoded, src_encoded, mask=tgt_src_mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = layer_norm(h + h_attn)
            
            # position-wise ffn SUBLAYER
            ffn_block = FFN(
                model_size=config.model_size,
                hidden_size=config.hidden_size,
                dropout_rate=config.dropout_rate,
            )
            # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            h_dense = ffn_block(h) # [b, n, model_size]
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = layer_norm(h + h_dense) # [b, n, model_size]
            
        return h
    


@dataclasses.dataclass
class Transformer(hk.Module):
    config: TransformerConfig
    is_training: bool
    
    def __post_init__(self):
        '''
        setup each components, embeddings is called on the fly
        '''
        
        super().__init__() # for dataclass decorated modules, this still should be called manually
        config = self.config
        
        key_size = config.model_size // config.num_heads
                
        self.encoder = Encoder(config=config)
        
        self.decoder = Decoder(config=config)
        
        self.embed = Embeddings(
            model_size=config.model_size,
            vocab_size=config.vocab_size,
        )
        
        def generator(x: jnp.ndarray): # weight sharing between embeddings and the final linear layer
            return jnp.dot(x, self.embed.embed.embeddings.T)
        
        self.generator = generator
        
    def __embed(
        self,
        x: jnp.ndarray, # sequence of token_ids, shape [b, n]
    ) -> jnp.ndarray:
        seq_len = x.shape[-1]
        pos_embed = hk.get_parameter("positional_embeddings", 
                                     [seq_len, self.config.model_size],
                                     init=PositionalEncoding())
        x_embed = self.embed(x) + pos_embed
        # we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
        x_embed = hk.dropout(hk.next_rng_key(), self.config.dropout_rate, x_embed) # [b, n, embed_size], embed_size == model_size
        return x_embed
    
    def encode(
        self,
        src_inputs: jnp.ndarray, # sequence of token_ids, shape [b, n], may contain padding token (usually 0)
    ) -> jnp.ndarray:
        dropout_rate = self.config.dropout_rate if self.is_training else 0
        
        # get combination of word & positional embedding
        src_inputs = src_inputs.astype('int32')
        src_embed = self.__embed(src_inputs) # shape [b, n] -> [b, n, embed_size], embed_size == model_size
        # _, seq_len, model_size = src_embed.shape
        # pos_embed = hk.get_parameter("positional_embeddings", [seq_len, model_size], init=PositionalEncoding())
        # src_embed += pos_embed
        # # we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
        # src_embed = hk.dropout(hk.next_rng_key(), dropout_rate, src_embed) # [b, n, embed_size], embed_size == model_size
        
        # get mask
        src_mask = jnp.einsum('bi, bj -> bij', src_inputs>0, src_inputs>0)[:, None, :] # shape [b, n] -> [b, n, n] -> [b, 1, n, n]
        
        return self.encoder(
            src_embed=src_embed,
            src_mask=src_mask,
            is_training=self.is_training,
        )
    
    def decode(
        self,
        tgt_inputs: jnp.ndarray, # sequence of token_ids, shape [b, n_q], may contain padding token (usually 0)
        src_inputs: jnp.ndarray, # sequence of token_ids, shape [b, n_k], to get mask
        src_encoded: jnp.ndarray, # shape [b, n_k, model_size]
    ) -> jnp.ndarray:
        dropout_rate = self.config.dropout_rate if self.is_training else 0
                
        # get combination of word & positional embedding
        tgt_inputs = tgt_inputs.astype('int32')        
        tgt_embed = self.__embed(tgt_inputs) # shape [b, n_q] -> [b, n_q, embed_size], embed_size == model_size
        # _, seq_len, model_size = tgt_embed.shape
        # pos_embed = hk.get_parameter("positional_embeddings", [seq_len, model_size], init=PositionalEncoding())
        # tgt_embed += pos_embed
        # # we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
        # tgt_embed = hk.dropout(hk.next_rng_key(), dropout_rate, tgt_embed) # [b, n_q, embed_size], embed_size == model_size
        
        # get masks
        tgt_src_mask = jnp.einsum('bi, bj -> bij', tgt_inputs>0, src_inputs>0)[:, None, :] # shape [b, n_q, n_k] -> [b, 1, n_q, n_k]
        tgt_mask = jnp.einsum('bi, bj -> bij', tgt_inputs>0, tgt_inputs>0)[:, None, :] # shape [b, n_q] -> [b, n_q, n_q] -> [b, 1, n_q, n_q]
        seq_len = tgt_embed.shape[-2]
        causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len))) # [1, 1, n_q, n_q]
        tgt_mask = tgt_mask * causal_mask # jnp.einsum('...i, ...i -> ...i', seq, causal_mask)
        
        return self.decoder(
            tgt_embed=tgt_embed,
            src_encoded=src_encoded,
            tgt_mask=tgt_mask,
            tgt_src_mask=tgt_src_mask,
            is_training=self.is_training,
        )
        
    def __call__(
        self,
        src_inputs: jnp.ndarray,
        tgt_inputs: jnp.ndarray,
        is_training: bool = True,
    ) -> jnp.ndarray:
        
        if is_training != self.is_training:
            self.is_training = is_training
        print(f'model is in {"training" if self.is_training else "evaluation"} mode.')
            
        encoded = self.encode(src_inputs) # shape [b, n, model_size]
        decoded = self.decode(tgt_inputs, src_inputs, encoded) # shape [b, n, model_size]
        # print(f"decoded shape: {decoded.shape}")
        
        return self.generator(decoded) # shape [b, n, vocab_size]
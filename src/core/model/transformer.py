"""
Created on Thu Sep 25 18:01:59 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import tensorflow as tf
from src.core.model.config import TransformerConfig

'''
###########################################
# Feed Forward Layer
###########################################
'''

@tf.keras.utils.register_keras_serializable(package="Custom")    
class FeedForwardBlock(tf.keras.layers.Layer):
    """
    Feed forward block for LLAMA model using TensorFlow/Keras.

    Args:
        dim (int): input dimension of the model.
        ffn_dim_multiplier (float): custom multiplier for hidden dimension of feed forward block.
        multiple_of (int): value to make hidden dimension of feed forward block multiple of.
    """

    def __init__(self, d_model, ffn_dim_multiplier=None, multiple_of=256):
        super().__init__()
        self.d_model = int(d_model)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = int(multiple_of)

        # Compute the hidden dimension for SwiGLU
        d_ff = 4 * d_model
        d_ff = int(2 * d_ff / 3)
        if ffn_dim_multiplier is not None:
            d_ff = int(d_ff * ffn_dim_multiplier)
        
        # Make it a multiple of `multiple_of`
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)

        # Define layers
        self.w1 = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.w2 = tf.keras.layers.Dense(d_model, use_bias=False)
        self.w3 = tf.keras.layers.Dense(d_ff, use_bias=False)

    def call(self, x):
        """
        Forward pass for the feed forward block.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply SwiGLU activation
        swish = tf.nn.silu(self.w1(x))      # (batch_size, seq_len, d_ff)
        x_v = self.w3(x)                    # (batch_size, seq_len, d_ff)
        x = swish * x_v                     # (batch_size, seq_len, d_ff)
        x = self.w2(x)                      # (batch_size, seq_len, d_model)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "multiple_of": self.multiple_of
        })
        return config
'''    
###########################################
# RMS Norm Layer
###########################################
'''

@tf.keras.utils.register_keras_serializable(package="Custom")    
class RMSNorm(tf.keras.layers.Layer):
    """
    RMSNorm (no bias): y = gamma * x / sqrt(mean(x^2) + eps)
    """
    def __init__(self, d_model, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

        self.weight = self.add_weight(
            name="weight",
            shape=(self.d_model,),
            initializer="ones",
            trainable=True,
        )

    def call(self, x):
        x2 = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        scale = tf.math.rsqrt(x2 + self.eps) # 1 / sqrt(mean(x^2)+eps)
        y = x * scale # normalize

        return self.weight * y

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "eps": self.eps}    

'''    
###########################################
# Grouped Query Attention Layer
###########################################
'''

# TF helper: cast-safe RoPE with dynamic shapes
def apply_rope_complex(x, freq_complex, start_pos=0):
    """
    x: [B, T, H, Dh] (any real dtype)
    freqs_cis: tf.complex64 [max_T, Dh//2]
    """
    b = tf.shape(x)[0]; t = tf.shape(x)[1]; h = tf.shape(x)[2]; d = tf.shape(x)[3]
    
    tf.debugging.assert_equal(d % 2, 0, message="head_dim must be even")
    half = d // 2
    x = tf.cast(x, tf.float32)
    x_pair = tf.reshape(x, (b, t, h, half, 2))
    x_comp = tf.complex(x_pair[...,0], x_pair[...,1])   # [B,T,H,half]

    freqs_slice = freq_complex[start_pos:start_pos + t]    # [T,half]
    freqs_bc = tf.expand_dims(tf.expand_dims(freqs_slice, 0), 2)  # [1,T,1,half]
    x_rot = x_comp * freqs_bc
    x_real = tf.math.real(x_rot); x_imag = tf.math.imag(x_rot)
    y = tf.stack([x_real, x_imag], axis=-1)
    y = tf.reshape(y, (b, t, h, d))
    
    return y

def attn_block(q, k, v, head_dim, mask):
    
    k = tf.cast(k, tf.float32)
    q = tf.cast(q, tf.float32)
    v = tf.cast(v, tf.float32)
    
    d = tf.cast(head_dim, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    Q = tf.transpose(q, [0,2,1,3])
    Kt= tf.transpose(k, [0,2,1,3])
    Vt= tf.transpose(v, [0,2,1,3])

    scores = tf.matmul(Q, Kt, transpose_b=True) / tf.sqrt(d)  # [B,H,T,T]
    
    # stable score
    scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
    
    if mask is not None:
        scores = scores + mask

    P = tf.nn.softmax(scores, axis=-1)
    
    out = tf.matmul(P, Vt)                                    # [B,H,T,D]
    
    return out

@tf.keras.utils.register_keras_serializable(package="Custom")
class GroupedQueryAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, n_kv_heads=None, use_cache=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.use_cache = use_cache
        
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads

        self.wq = tf.keras.layers.Dense(self.n_heads * self.head_dim, use_bias=False)
        self.wk = tf.keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.wv = tf.keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.wo = tf.keras.layers.Dense(self.d_model, use_bias=False)

        self.cache_k = None
        self.cache_v = None
        self.max_seq_len = None

    def build(self, input_shape):
        super().build(input_shape)

    def _init_cache(self, batch_size, max_seq_len):
        if not self.use_cache:
            return
        if (self.cache_k is None) or (self.max_seq_len is None) or (self.max_seq_len < max_seq_len):
            self.cache_k = self.add_weight(
                name="cache_k",
                shape=(batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
                initializer="zeros",
                trainable=False
            )
            self.cache_v = self.add_weight(
                name="cache_v",
                shape=(batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
                initializer="zeros",
                trainable=False
            )
            self.max_seq_len = int(max_seq_len)

    def call(self, x, freq_complex=None, start_pos=0, attn_mask=None, training=False):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]
     
        q = tf.reshape(self.wq(x), (B, T, self.n_heads, self.head_dim))
        k = tf.reshape(self.wk(x), (B, T, self.n_kv_heads, self.head_dim))
        v = tf.reshape(self.wv(x), (B, T, self.n_kv_heads, self.head_dim))

        # RoPE complex (kept in float32 for stability)
        if freq_complex is not None:
            q = apply_rope_complex(q, freq_complex, start_pos)
            k = apply_rope_complex(k, freq_complex, start_pos)

        # cache (store in fp32)
        if self.use_cache:
            max_len = tf.shape(freq_complex)[0] if freq_complex is not None else T
            
            self._init_cache(batch_size=tf.shape(x)[0], max_seq_len=max_len)
            
            self.cache_k[:, start_pos:start_pos + T].assign(k, tf.float32)
            self.cache_v[:, start_pos:start_pos + T].assign(v, tf.float32)
            
            k = self.cache_k[:, :start_pos + T]
            v = self.cache_v[:, :start_pos + T]
            
        # Repeat KV for GQA
        if self.n_kv_heads != self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = tf.repeat(k, repeats=reps, axis=2)
            v = tf.repeat(v, repeats=reps, axis=2)

        # [B,H,T,D]
        out = attn_block(q, k, v, self.head_dim, attn_mask)

        out = tf.transpose(out, [0,2,1,3])
        out = tf.reshape(out, (B, T, self.d_model))

        return self.wo(out)

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "use_cache": self.use_cache,
        }
    
'''    
###########################################
# Transformer Block Layer
###########################################
'''
    
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    """
    LLaMA-style Transformer Block (RMSNorm -> GQA+RoPE -> residual -> RMSNorm -> SwiGLU FFN -> residual)
    - No dependency on fixed batch_size/seq_len.
    - Passes RoPE freqs and start_pos to attention.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads=None,
        ffn_dim_multiplier=None,
        multiple_of=256,
        use_cache=False,
        norm_eps=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.ffn_dim_multiplier = None if ffn_dim_multiplier is None else ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.use_cache = use_cache
        self.norm_eps = norm_eps

        # Attention
        self.attention = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            use_cache=self.use_cache,
        )

        # FeedForward
        self.feed_forward = FeedForwardBlock(
            d_model=self.d_model,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
            multiple_of=self.multiple_of,
        )

        # Norms
        self.attention_norm = RMSNorm(self.d_model, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.d_model, eps=self.norm_eps)
        
    def call(self, x, start_pos=0, freq_complex=None, attn_mask=None, training=False):
        """
        x: [B, T, d_model]
        start_pos: int (for streaming/cache; 0 for full-seq training)
        freq_complex: complex64 RoPE frequencies [max_T, head_dim//2] (optional but recommended)
        attn_mask: additive attn_mask broadcastable to [B, H, T_q, T_k] (e.g., causal + padding)
        """
        # Attention block
        h = self.attention_norm(x)
        h = self.attention(h, freq_complex=freq_complex, start_pos=start_pos, attn_mask=attn_mask, training=training)
        x = x + h  # residual

        # Feed-forward block
        h2 = self.ffn_norm(x)
        h2 = self.feed_forward(h2)
        x = x + h2  # residual

        return x

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "multiple_of": self.multiple_of,
            "use_cache": self.use_cache,
            "norm_eps": self.norm_eps
        }

'''    
###########################################
# Transformer model
###########################################
'''

def causal_mask(T, dtype=tf.float32):
    """
    Returns an additive causal mask shaped [1, 1, T, T].
    0.0 on/below diagonal (allowed), -inf above (blocked).
    """
    i = tf.range(T)[:, None]
    j = tf.range(T)[None, :]
    upper = j > i  # True where j (key) is strictly in the future of i (query)
    neg_inf = tf.constant(-1e9, dtype=dtype if dtype.is_floating else tf.float32)
    mask = tf.where(upper, neg_inf, tf.zeros([], dtype=neg_inf.dtype))  # [T, T]
    return mask[None, None, :, :]  # [1, 1, T, T]
    
def precompute_theta_pos_freqs(head_dim, seq_len, theta=10000.0):
    """
    Precomputes the rotary positional embeddings as complex numbers.

    Args:
        head_dim (int): The dimension of each head.
        seq_len (int): The maximum sequence length.
        theta (float): The scaling factor for rotary frequencies.

    Returns:
        tf.Tensor: Complex tensor of shape (seq_len, head_dim // 2).
    """
    # 1. Generate frequency indices
    theta_num = tf.range(0, head_dim, 2, dtype=tf.float32) # [head_dim / 2]
    theta_freqs = 1.0 / (theta ** (theta_num / head_dim)) # [head_dim / 2]

    # 2. Generate positional indices
    positions = tf.range(seq_len, dtype=tf.float32) # [seq_len]

    # 3. Compute the outer product to get the rotational angles
    m_theta = tf.tensordot(positions, theta_freqs, axes=0) # [seq_len, head_dim / 2]

    # 4. Create the complex representation (cos, sin)
    cos_pos = tf.cos(m_theta)
    sin_pos = tf.sin(m_theta)

    # 5. **Create complex representation properly**
    freq_complex = tf.complex(cos_pos, sin_pos)  # [seq_len, head_dim / 2]
    
    return freq_complex
    
@tf.keras.utils.register_keras_serializable(package="Custom")    
class Transformer(tf.keras.Model):
    """
    LLaMA-style Transformer:
      token_embed -> [x + Attn( RMSNorm(x) )] * L -> [x + FFN( RMSNorm(x) )] * L
      -> RMSNorm -> logits
    """
    def __init__(self, config: "TransformerConfig", **kwargs):
        super().__init__(**kwargs)

        self.cfg = config  # single source of truth
        
        # Convenience fields (optional)
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.ffn_dim_multiplier = config.ffn_dim_multiplier
        self.multiple_of = config.multiple_of
        self.seq_len = config.seq_len
        self.norm_eps = config.norm_eps
        self.use_cache = config.use_cache
        
        # Embedding
        self.tok_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)

        self.blocks = tuple(
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                ffn_dim_multiplier=self.ffn_dim_multiplier,
                multiple_of=self.multiple_of,
                use_cache=self.use_cache,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.n_layers)
        )
        
        # Final norm
        self.norm = RMSNorm(self.d_model, eps=self.norm_eps)

        # Output projection (untied; tie if you want)
        self.projection_layer = tf.keras.layers.Dense(self.vocab_size, use_bias=False)

        # RoPE cache (TF complex)
        self.freq_complex = precompute_theta_pos_freqs(self.d_model // self.n_heads, self.seq_len * 2)
        
    def call(self, x, start_pos=0, training=False):
        """
        Causal + pad-safe masking.
        - Keys: pad + causal are masked as before.
        - Queries: PAD queries are zeroed so they don't contribute.
        - All-masked rows: neutralized so softmax can't produce NaN.
        """
        
        T = tf.shape(x)[1] # seq len
        
        # 1) Token embeddings
        h = self.tok_embeddings(x)  # [B,T,dim]
        
        # 2) RoPE slice
        if self.use_cache:
            freq_complex = self.freq_complex[start_pos:start_pos + T]
        else:
            freq_complex = self.freq_complex[:T]
            
        # 3) Simple causal mask (broadcastable to [B, n_heads, T, T])
        attn_mask = causal_mask(T, dtype=h.dtype)

        # 3) Execute forward pass
        for block in self.blocks:
            h = block(h, 
                      start_pos=start_pos, 
                      freq_complex=freq_complex, 
                      attn_mask=attn_mask, 
                      training=training)

        # 4) Normalize hidden state
        h = self.norm(h)
        
        # 5) Output projection
        logits = self.projection_layer(h)  # [B,T,vocab]
        
        return logits
    
    def get_config(self):
        base = super().get_config()
        base.update({"config": self.cfg.to_dict()})
        return base

    @classmethod
    def from_config(cls, config, custom_objects=None):
        cfg_dict = config.pop("config")
        cfg = TransformerConfig.from_dict(cfg_dict)
        return cls(config=cfg, **config)
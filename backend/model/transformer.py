import tensorflow as tf


# =========================
# Positional Encoding
# =========================
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)

        x = self.token_emb(x)
        return x + positions


# =========================
# Causal Mask
# =========================
def create_causal_mask(seq_len):
    """
    Prevents attention to future tokens
    Shape: (1, 1, seq_len, seq_len)
    """
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask


# =========================
# Multi-Head Self Attention
# =========================
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_dense = tf.keras.layers.Dense(embed_dim * 3)
        self.out_dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        qkv = self.qkv_dense(x)
        qkv = tf.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Apply causal mask
        mask = create_causal_mask(seq_len)
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))

        scores = scores * mask + (1.0 - mask) * (-1e9)

        attention = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(attention, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.embed_dim))

        return self.out_dense(out)


# =========================
# Feed Forward Network
# =========================
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_dim),
        ])

    def call(self, x):
        return self.net(x)


# =========================
# Transformer Block
# =========================
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.attn(x)
        attn_output = self.dropout1(attn_output, training=training)

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.norm2(x + ffn_output)


# =========================
# IIMo Transformer Model
# =========================
class IIMoModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=512,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = PositionalEmbedding(
            vocab_size, hidden_size, max_seq_len
        )

        self.transformer_blocks = [
            TransformerBlock(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.lm_head = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        """
        inputs: (batch_size, seq_len)
        """
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids, max_length=100):
        """
        Greedy decoding for inference
        """
        for _ in range(max_length):
            logits = self(input_ids, training=False)

            next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, axis=1)

            input_ids = tf.concat([input_ids, next_token], axis=1)

        return input_ids
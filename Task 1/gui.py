import tkinter as tk
import tensorflow as tf
from tensorflow import keras
from keras import layers
import json
import re
import string
import numpy as np

# ======== Transformer Components ========

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        return tf.tile(mask, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
        })
        return config

# ======== Load vectorizers & vocab ========

def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

tamil_vocab = load_json("tam_vocab.json")
french_vocab = load_json("fre_vocab.json")

french_vec_conf = load_json("fre_vectorization_config.json")
french_vec_conf["max_tokens"] = max(french_vec_conf.get("max_tokens", 15000), len(french_vocab) + 10)
french_vectorizer = tf.keras.layers.TextVectorization.from_config(french_vec_conf)
french_vectorizer.set_vocabulary(french_vocab)

tamil_vec_conf = load_json("tam_vectorization_config.json")
tamil_vec_conf["max_tokens"] = len(tamil_vocab)
tamil_vectorizer = tf.keras.layers.TextVectorization.from_config(tamil_vec_conf)
tamil_vectorizer.set_vocabulary(tamil_vocab)

idx_to_tamil = dict(enumerate(tamil_vocab))

# ======== Load Model ========

translator_model = keras.models.load_model(
    "french_tamil_transformer.keras",
    custom_objects={
        "PositionalEmbedding": TokenAndPositionEmbedding,
        "TransformerEncoder": EncoderBlock,
        "TransformerDecoder": DecoderBlock
    },
    compile=False
)
translator_model.load_weights("french_tamil_weights.weights.h5")

# ======== Translation Logic ========

PUNCT_TO_STRIP = string.punctuation.replace("[", "").replace("]", "") + "¿"
MAX_OUTPUT_TOKENS = 20

def preprocess_text(txt):
    txt = txt.lower()
    return re.sub(f"[{re.escape(PUNCT_TO_STRIP)}]", "", txt)

def translate_french_to_tamil(sentence):
    sentence = preprocess_text(sentence)
    enc_input = french_vectorizer([sentence])
    output_seq = "[start]"
    for _ in range(MAX_OUTPUT_TOKENS):
        dec_input = tamil_vectorizer([output_seq])[:, :-1]
        preds = translator_model([enc_input, dec_input])
        next_id = tf.argmax(preds[0, _, :]).numpy().item()
        next_token = idx_to_tamil.get(next_id, "")
        output_seq += " " + next_token
        if next_token == "[end]":
            break
    return output_seq.replace("[start]", "").replace("[end]", "").strip()

# ======== Tkinter GUI ========

def on_translate():
    word = word_entry.get().strip()
    if len(word) != 5:
        result_label.config(text="Please enter exactly 5 letters in French.")
        return
    try:
        translation = translate_french_to_tamil(word)
        result_label.config(text=f"Tamil Translation: {translation}")
    except Exception as err:
        result_label.config(text=f"Error: {err}")

root = tk.Tk()
root.title("French ➜ Tamil Translator")

tk.Label(root, text="Enter a 5-letter French word:").pack(pady=10)
word_entry = tk.Entry(root, width=30, font=("Arial", 14))
word_entry.pack(pady=5)

tk.Button(root, text="Translate", command=on_translate).pack(pady=5)
result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

root.mainloop()

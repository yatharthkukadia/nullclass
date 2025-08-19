import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import json
import pickle
import tensorflow as tf
from tkinter import messagebox, ttk

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        decoder_output, encoder_output = inputs

        score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, encoder_output)
        return context_vector

    def get_config(self):
        return super().get_config()

french_model = load_model('english_to_french_model')
hindi_model = load_model('eng_hi_translation_model.keras', custom_objects={'AttentionLayer': AttentionLayer})

with open('english_tokenizer.json', 'rb') as f:
    eng_tokenizer_french = json.load(f)
with open('french_tokenizer.json', 'rb') as f:
    french_tokenizer = json.load(f)
with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_len_french = json.load(f)
with open('input_tokenizer.pkl', 'rb') as f:
    eng_tokenizer_hindi = pickle.load(f)
with open('target_tokenizer.pkl', 'rb') as f:
    hindi_tokenizer = pickle.load(f)

max_len_french = 21  # Replace with actual value, e.g., 20
max_len_hindi = 50  # Replace with actual value, e.g., 50
max_len_hindi_decoder = max_len_hindi - 1

def preprocess_french_input(sentence, eng_tokenizer, max_len):
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded


def decode_french_sequence(prediction, tokenizer):
    index_to_word = {i: w for w, i in tokenizer.word_index.items()}
    index_to_word[0] = '<PAD>'
    return ' '.join([index_to_word[np.argmax(vector)] for vector in prediction]).replace('<PAD>', '').strip()


def preprocess_hindi_input(sentence, eng_tokenizer, hindi_tokenizer, max_len_enc, max_len_dec):
    input_seq = eng_tokenizer.texts_to_sequences([sentence])
    input_padded = pad_sequences(input_seq, maxlen=max_len_enc, padding='post')
    decoder_input = np.zeros((1, max_len_dec))
    decoder_input[0, 0] = hindi_tokenizer.word_index.get('<sos>', 0)
    return input_padded, decoder_input

def decode_hindi_sequence(model, input_seq, decoder_input, hindi_tokenizer, max_len_dec):
    prediction = model.predict([input_seq, decoder_input], verbose=0)
    index_to_word = {i: w for w, i in hindi_tokenizer.word_index.items()}
    index_to_word[0] = '<PAD>'
    translated = []
    for vector in prediction[0]:
        word_idx = np.argmax(vector)
        word = index_to_word.get(word_idx, '')
        if word == '<eos>':
            break
        if word and word != '<PAD>':
            translated.append(word)
    return ' '.join(translated).strip()

def translate_sentence():
    english_input = input_text.get("1.0", tk.END).strip()
    
    # Count letters (excluding spaces and punctuation)
    letter_count = sum(c.isalpha() for c in english_input)
    if letter_count < 10:
        messagebox.showwarning("Invalid Input", "Input must have 10 or more letters. Please upload again.")
        output_text.delete("1.0", tk.END)
        return

    selected_language = language_var.get()
    if selected_language == "French":
        input_seq = preprocess_french_input(english_input, eng_tokenizer_french, max_len_french)
        prediction = french_model.predict(input_seq, verbose=0)
        translation = decode_french_sequence(prediction[0], french_tokenizer)
    else:  # Hindi
        input_seq, decoder_input = preprocess_hindi_input(
            english_input, eng_tokenizer_hindi, hindi_tokenizer, max_len_hindi, max_len_hindi_decoder
        )
        translation = decode_hindi_sequence(
            hindi_model, input_seq, decoder_input, hindi_tokenizer, max_len_hindi_decoder
        )

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, translation)
root = tk.Tk()
root.title("Language Translator")
root.geometry("600x400")

# Input section
tk.Label(root, text="Enter the text to be translated", font=("Arial", 12)).pack(pady=10)
input_text = tk.Text(root, height=5, width=50, font=("Arial", 10))
input_text.pack(pady=5)

# Language selection
tk.Label(root, text="Select the language to translate to", font=("Arial", 12)).pack(pady=5)
language_var = tk.StringVar(value="French")  # Default selection
language_dropdown = ttk.Combobox(root, textvariable=language_var, values=["French", "Hindi"], state="readonly")
language_dropdown.pack(pady=5)

# Translate button
translate_button = tk.Button(root, text="Translate", command=translate_sentence, font=("Arial", 12), bg="#4CAF50", fg="white")
translate_button.pack(pady=10)

# Output section
tk.Label(root, text="Translation:", font=("Arial", 12)).pack(pady=5)
output_text = tk.Text(root, height=5, width=50, font=("Arial", 10))
output_text.pack(pady=5)

# Start the Tkinter event loop
root.mainloop()
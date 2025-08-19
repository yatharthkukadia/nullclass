import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
from datetime import datetime
import tensorflow as tf

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

    

    

# Load model and tokenizers
model = load_model('eng_hi_translation_model.keras', custom_objects={'AttentionLayer': AttentionLayer})

with open('input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)

with open('target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

# Reverse target tokenizer
reverse_target_index = {i: w for w, i in target_tokenizer.word_index.items()}
reverse_target_index[0] = ''

# Parameters
max_encoder_len = 50
max_decoder_len = 50  # including <sos> and <eos>

def starts_with_vowel(word):
    return word[0].lower() in 'aeiou'

def is_vowel_time_allowed():
    current_time = datetime.now().time()
    return current_time.hour == 21  # 9 PM

def decode_sequence(input_seq):
    # Prepare encoder input
    input_seq = input_tokenizer.texts_to_sequences([input_seq])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_len, padding='post')

    # Prepare initial decoder input with <sos>
    target_seq = np.zeros((1, max_decoder_len - 1))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']

    decoded_sentence = []

    for i in range(1, max_decoder_len - 1):
        output_tokens = model.predict([input_seq, target_seq], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, i - 1, :])
        sampled_word = reverse_target_index.get(sampled_token_index, '')

        if sampled_word == '<eos>' or sampled_word == '':
            break

        decoded_sentence.append(sampled_word)
        target_seq[0, i] = sampled_token_index

    return ' '.join(decoded_sentence)


def translate():
    eng_word = entry.get().strip()

    if not eng_word:
        messagebox.showerror("Error", "Please enter an English word.")
        return

    if starts_with_vowel(eng_word) and not is_vowel_time_allowed():
        messagebox.showerror("Vowel Error", "This word starts with a vowel. Please provide another word.")
        return

    try:
        translation = decode_sequence(eng_word)
        output_label.config(text=f"Hindi: {translation}")
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("English to Hindi Translator")
root.geometry("400x250")
root.config(padx=20, pady=20)

tk.Label(root, text="Enter English Word:", font=('Arial', 12)).pack(pady=10)
entry = tk.Entry(root, font=('Arial', 12), width=30)
entry.pack()

tk.Button(root, text="Translate", font=('Arial', 12), command=translate).pack(pady=10)
output_label = tk.Label(root, text="", font=('Arial', 12), fg="green")
output_label.pack(pady=20)

root.mainloop()

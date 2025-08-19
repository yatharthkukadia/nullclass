import tkinter as tk 
import datetime
import speech_recognition as sr
import numpy as np

from keras.models import load_model
import tensorflow as tf
from keras.utils import pad_sequences
import pickle

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

    
    

model = load_model('eng_hi_translation_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
with open('input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)
with open('target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

index_to_word = {v: k for k, v in target_tokenizer.word_index.items()}
index_to_word[0] = ''

def preprocess_input(sentence, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded

def decode_sequence(input_seq, model, decoder_input_len):
    target_seq = np.zeros((1, decoder_input_len))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']
    
    output_sentence = []

    for i in range(1, decoder_input_len):
        prediction = model.predict([input_seq, target_seq], verbose=0)
        predicted_id = np.argmax(prediction[0, i-1, :])

        if index_to_word.get(predicted_id, '') == '<eos>':
            break

        output_sentence.append(index_to_word.get(predicted_id, ''))
        target_seq[0, i] = predicted_id

    return ' '.join(output_sentence)

eng_max_len = 50  
hi_max_len = 50

def translate_audio():
    current_time = datetime.datetime.now().time()
    start = datetime.time(21, 30)
    end = datetime.time(22, 0)

    if not (start <= current_time <= end):
        output_label.config(text="Taking rest, see you tomorrow!")
        return

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        output_label.config(text="Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            input_text_label.config(text=f"English: {text}")
            input_seq = preprocess_input(text)
            translation = decode_sequence(input_seq)
            output_label.config(text=f"Hindi: {translation}")
        except sr.UnknownValueError:
            output_label.config(text="Please repeat! I couldn’t understand.")
        except sr.RequestError:
            output_label.config(text="Speech Recognition service error.")
        except Exception as e:
            output_label.config(text=f"Error: {str(e)}")


root = tk.Tk()
root.title("Voice Translator")
root.geometry("500x300")

record_btn = tk.Button(root, text="🎤 Speak", font=("Arial", 14), command=translate_audio)
record_btn.pack(pady=20)

input_text_label = tk.Label(root, text="", font=("Arial", 12))
input_text_label.pack()

output_label = tk.Label(root, text="", font=("Arial", 14), wraplength=400, justify="center")
output_label.pack(pady=10)

root.mainloop()
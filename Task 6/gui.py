import tkinter as tk
import speech_recognition as sr
import pyttsx3
import tensorflow as tf
import json
from keras.models import load_model
import numpy as np

# Load models
eng_to_spa_model = load_model("transformer_model")
spa_to_eng_model = load_model("transformer_model_spa_to_eng")

# Load vocab/tokenizer configs
with open("eng_vocab.json", "r", encoding="utf-8") as f:
    eng_vocab = json.load(f)
with open("spa_vocab.json", "r", encoding="utf-8") as f:
    spa_vocab = json.load(f)

# Simple index <-> token mapping
eng_word_index = {word: i for i, word in enumerate(eng_vocab)}
spa_word_index = {word: i for i, word in enumerate(spa_vocab)}
inv_eng_vocab = {i: word for i, word in enumerate(eng_vocab)}
inv_spa_vocab = {i: word for i, word in enumerate(spa_vocab)}

# Speech recognizer + TTS engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Dummy translate function (replace with your inference loop)
def dummy_translate(text, mode="eng_to_spa"):
    if mode == "eng_to_spa":
        return "[spa translation of]: " + text
    else:
        return "[eng translation of]: " + text

def speak(text, lang="en"):
    voices = engine.getProperty("voices")
    if lang == "en":
        engine.setProperty("voice", voices[0].id)  # English voice
    else:
        engine.setProperty("voice", voices[1].id)  # Spanish voice
    engine.say(text)
    engine.runAndWait()

def record_and_translate(mode):
    with sr.Microphone() as source:
        status_label.config(text="Listening...")
        root.update()
        audio = recognizer.listen(source)
    try:
        if mode == "eng_to_spa":
            spoken_text = recognizer.recognize_google(audio, language="en")
            translation = dummy_translate(spoken_text, "eng_to_spa")
            eng_text_box.delete("1.0", tk.END)
            eng_text_box.insert(tk.END, spoken_text)
            spa_text_box.delete("1.0", tk.END)
            spa_text_box.insert(tk.END, translation)
            speak(translation, lang="es")

        else:  # Spanish to English
            spoken_text = recognizer.recognize_google(audio, language="es")
            translation = dummy_translate(spoken_text, "spa_to_eng")
            spa_text_box.delete("1.0", tk.END)
            spa_text_box.insert(tk.END, spoken_text)
            eng_text_box.delete("1.0", tk.END)
            eng_text_box.insert(tk.END, translation)
            speak(translation, lang="en")

        status_label.config(text="Done")

    except Exception as e:
        status_label.config(text="Error: " + str(e))

# GUI
root = tk.Tk()
root.title("Realtime English ↔ Spanish Voice Translator")

tk.Label(root, text="English").grid(row=0, column=0)
tk.Label(root, text="Spanish").grid(row=0, column=1)

eng_text_box = tk.Text(root, height=5, width=40)
eng_text_box.grid(row=1, column=0)
spa_text_box = tk.Text(root, height=5, width=40)
spa_text_box.grid(row=1, column=1)

btn_eng = tk.Button(root, text="🎤 Speak English", command=lambda: record_and_translate("eng_to_spa"))
btn_eng.grid(row=2, column=0, pady=10)

btn_spa = tk.Button(root, text="🎤 Speak Spanish", command=lambda: record_and_translate("spa_to_eng"))
btn_spa.grid(row=2, column=1, pady=10)

status_label = tk.Label(root, text="Status: Ready", fg="blue")
status_label.grid(row=3, columnspan=2)

root.mainloop()

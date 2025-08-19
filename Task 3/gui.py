import cv2
import pytesseract
import json
from PIL import Image
from tkinter import Tk, filedialog, Button, Text, Label, END, Scrollbar, RIGHT, Y, Frame
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import os

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model and tokenizers
model = load_model("english_to_french_model")
with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    eng_token = tokenizer_from_json(json.load(f))
with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_token = tokenizer_from_json(json.load(f))
with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_len_french = json.load(f)

max_len = max_len_french  # use your actual max_len from training

# Image OCR
def image_loader(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

# Video OCR
def video_loader(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    if ret:
        text = pytesseract.image_to_string(frame, lang='eng')
        return text
    return ""

# Preprocess
def preprocess_input(sentence, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded

# Decode model output
def decode_sequence(prediction, french_tokenizer):
    index_to_word = {i: w for w, i in french_tokenizer.word_index.items()}
    index_to_word[0] = '<PAD>'
    return ' '.join([index_to_word.get(np.argmax(vector), '') for vector in prediction])

# Translate sentence
def translate_sentence(sentence, model, eng_tokenizer, french_tokenizer, max_len):
    input_seq = preprocess_input(sentence, eng_tokenizer, max_len)
    prediction = model.predict(input_seq)
    return decode_sequence(prediction[0], french_tokenizer)

# GUI function
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.png *.jpeg *.mp4")])
    if not file_path:
        return

    # Display selected path
    status_label.config(text=f"Selected: {os.path.basename(file_path)}")

    if file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        extracted_text = image_loader(file_path)
    elif file_path.lower().endswith('.mp4'):
        extracted_text = video_loader(file_path)
    else:
        extracted_text = "Unsupported file type."

    translated = translate_sentence(extracted_text, model, eng_token, french_token, max_len)

    # Output
    output_text.delete(1.0, END)
    output_text.insert(END, f"Extracted Text:\n{extracted_text}\n\n")
    output_text.insert(END, f"Translated Text (French):\n{translated}\n")

# Tkinter GUI setup
app = Tk()
app.title("Image/Video Translator (Eng ➜ Fr)")
app.geometry("700x500")

status_label = Label(app, text="Choose an image or video file...", font=("Arial", 12))
status_label.pack(pady=10)

choose_button = Button(app, text="Upload File", command=open_file, font=("Arial", 12))
choose_button.pack(pady=5)

frame = Frame(app)
frame.pack(expand=True, fill='both')

scrollbar = Scrollbar(frame)
scrollbar.pack(side=RIGHT, fill=Y)

output_text = Text(frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 11))
output_text.pack(expand=True, fill='both')
scrollbar.config(command=output_text.yview)

app.mainloop()

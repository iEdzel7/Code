from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import pytesseract
from PIL import Image
import easyocr

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load trained readability model
model = tf.keras.models.load_model("code_readability_model.h5")

# Load bug localization model
bug_model = tf.keras.models.load_model("bug_localization_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("BugTokenizer.pkl", "rb") as handle:
    bug_tokenizer = pickle.load(handle)

# Tokenization settings
max_len = 200
bug_max_len = 500

# Create the uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_readability(java_code):
    """Predict numerical readability score using the trained ML model."""
    sequence = tokenizer.texts_to_sequences([java_code])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    score = round(prediction * 4 + 1)
    return score

def predict_bug_status_ml(java_code):
    """Predict if a Java code snippet is Buggy or Non-Buggy using ML model."""
    sequence = bug_tokenizer.texts_to_sequences([java_code])
    padded_sequence = pad_sequences(sequence, maxlen=bug_max_len, padding="post")
    prediction = bug_model.predict(padded_sequence)[0][0]
    return "Buggy Code Detected!" if prediction > 0.5 else "Code is Non-Buggy!"

def estimate_time_complexity(java_code):
    """Estimate the time complexity of Java code using static pattern matching."""
    loop_count = len(re.findall(r'\b(for|while)\s*\(', java_code))
    nested_loops = len(re.findall(r'for\s*\(.*\)\s*\{[\s\S]*for\s*\(.*\)', java_code))
    recursion_count = len(re.findall(r'\bpublic\s+\w+\s*\(.*\)\s*\{[\s\S]*\breturn\s+\w+\(.*\);', java_code))
    if recursion_count > 0:
        return "O(2^N) (Exponential - Recursion Detected)"
    elif nested_loops > 0:
        return "O(N^2) (Quadratic - Nested Loops Detected)"
    elif loop_count > 0:
        return "O(N) (Linear - Single Loop Detected)"
    elif "if" in java_code:
        return "O(log N) (Logarithmic - Conditional Statements Present)"
    else:
        return "O(1) (Constant - No Loops or Recursion)"

def estimate_space_complexity(java_code):
    """Estimate the space complexity of Java code using static pattern matching."""
    array_count = len(re.findall(r'\b(int|double|float|char|boolean|String)\s*\[\]', java_code))
    list_count = len(re.findall(r'ArrayList|LinkedList|List<', java_code))
    map_count = len(re.findall(r'HashMap|TreeMap|Map<', java_code))
    object_count = len(re.findall(r'new\s+[A-Z]\w*\(', java_code))
    recursion_count = len(re.findall(r'\bpublic\s+\w+\s*\(.*\)\s*\{[\s\S]*\breturn\s+\w+\(.*\);', java_code))
    if recursion_count > 0:
        return "O(N) (Recursive Call Stack Detected)"
    elif array_count > 0 or list_count > 0 or map_count > 0 or object_count > 0:
        return "O(N) (Dynamic Memory Allocation Detected)"
    else:
        return "O(1) (Constant - Primitive Variables Only)"

reader = easyocr.Reader(['en'])

@app.route("/")
def index():
    return render_template("codelytics.html")

@app.route("/image")
def image_page():
    return render_template("image.html")

@app.route("/text")
def text_page():
    return render_template("text.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    java_code = data.get("code", "")
    numerical_score = predict_readability(java_code)
    final_bug_status = predict_bug_status_ml(java_code)
    complexity_analysis = estimate_time_complexity(java_code)
    space_complexity = estimate_space_complexity(java_code)
    return jsonify({
        "score": numerical_score,
        "bug_status": final_bug_status,
        "complexity_analysis": complexity_analysis,
        "space_complexity": space_complexity
    })

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        return None

def save_text_to_file(text, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"message": "No file uploaded."}), 400
    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        return jsonify({"message": "Failed to extract text from the image."}), 400
    text_filename = image.filename.rsplit(".", 1)[0] + ".txt"
    save_text_to_file(extracted_text, text_filename)
    numerical_score = predict_readability(extracted_text)
    final_bug_status = predict_bug_status_ml(extracted_text)
    complexity_analysis = estimate_time_complexity(extracted_text)
    space_complexity = estimate_space_complexity(extracted_text)
    return jsonify({
        "message": "Image uploaded and processed successfully.",
        "extracted_code": extracted_text,
        "score": numerical_score,
        "bug_status": final_bug_status,
        "complexity_analysis": complexity_analysis,
        "space_complexity": space_complexity
    })

if __name__ == "__main__":
    app.run(debug=True)

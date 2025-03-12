from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import groq
import re
import os
import pytesseract
from PIL import Image
import easyocr
import requests
import io

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable cross-origin requests

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

def classify_readability_with_llama(java_code):
    """Use Groq's Llama 3 AI to classify code and assign a readability/efficiency score."""
    client = groq.Client(api_key="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")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": 
                "Analyze the following Java code snippet and provide a readability/efficiency score from 1-5. "
                "Score the code **1 or 2** if it is difficult to read due to poor formatting, unclear variable names, lack of indentation, or compressed structure. "
                "Score the code **4 or 5** if it follows good coding practices such as proper indentation, clear variable names, and spacing. "
                "If the code is average but could use minor improvements, assign a score of **3**. "
                "Return only the number (1, 2, 3, 4, or 5). Do not include any other text."
            },
            {"role": "user", "content": java_code}
        ]
    )

    llama_score_text = response.choices[0].message.content.strip()
    try:
        llama_score = int(llama_score_text)
        if 1 <= llama_score <= 5:
            return llama_score
    except ValueError:
        return None

def compute_final_score(ml_score, llama_score):
    """Combine the ML model score and Llama 3 AI score with a weighted approach."""
    if llama_score is None:
        return ml_score  

    final_score = round((ml_score * 0.4) + (llama_score * 0.6))
    return max(1, min(final_score, 5))

### ---- Bug Localization ---- ###
def predict_bug_status_ml(java_code):
    """Predict if a Java code snippet is Buggy or Non-Buggy using ML model."""
    sequence = bug_tokenizer.texts_to_sequences([java_code])
    padded_sequence = pad_sequences(sequence, maxlen=bug_max_len, padding="post")

    prediction = bug_model.predict(padded_sequence)[0][0]
    return prediction  # Higher values mean more likely buggy

def classify_bug_with_groq(java_code):
    client = groq.Client(api_key="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")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": 
                "Analyze the following Java code snippet and determine if it is buggy. "
                "If the code has missing conditions, runtime errors, poor structure, or seems logically incorrect, classify it as **buggy**. "
                "If it follows logical patterns and appears correct, classify it as **non-buggy**. "
                "Return only a probability score between **0 and 1**, where: "
                "0 = Absolutely non-buggy, 1 = Absolutely buggy. "
                "Do NOT include any explanation or text, only return a valid floating-point number."
            },
            {"role": "user", "content": java_code}
        ]
    )

    groq_score_text = response.choices[0].message.content.strip()

    try:
        # Ensure the response is a valid float between 0 and 1
        groq_score = float(groq_score_text)
        if 0 <= groq_score <= 1:
            return groq_score
        else:
            print(f"Returned an out-of-range score: {groq_score}")
            return None  # Return None if out of range
    except ValueError:
        print(f"Response invalid: {groq_score_text}")
        return None  # Return None if Groq response is invalid


def compute_final_bug_score(ml_score, groq_score):
    if groq_score is None:
        return "Response invalid. ML Score: " + ("Buggy" if ml_score > 0.5 else "Non-Buggy")

    final_score = (ml_score * 0.35) + (groq_score * 0.65)
    return "Buggy Code Detected!" if final_score > 0.5 else "Code is Non-Buggy!"


def estimate_time_complexity(java_code):
    """Estimate the time complexity of Java code using static pattern matching."""
    
    # Count occurrences of common patterns
    loop_count = len(re.findall(r'\b(for|while)\s*\(', java_code))
    nested_loops = len(re.findall(r'for\s*\(.*\)\s*\{[\s\S]*for\s*\(.*\)', java_code))
    recursion_count = len(re.findall(r'\bpublic\s+\w+\s*\(.*\)\s*\{[\s\S]*\breturn\s+\w+\(.*\);', java_code))

    # Decision logic for complexity estimation
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
    
    # Detect arrays, lists, hashmaps, objects
    array_count = len(re.findall(r'\b(int|double|float|char|boolean|String)\s*\[\]', java_code))
    list_count = len(re.findall(r'ArrayList|LinkedList|List<', java_code))
    map_count = len(re.findall(r'HashMap|TreeMap|Map<', java_code))
    object_count = len(re.findall(r'new\s+[A-Z]\w*\(', java_code))
    
    # Detect recursion
    recursion_count = len(re.findall(r'\bpublic\s+\w+\s*\(.*\)\s*\{[\s\S]*\breturn\s+\w+\(.*\);', java_code))

    # Decision logic for space complexity estimation
    if recursion_count > 0:
        return "O(N) (Recursive Call Stack Detected)"
    elif array_count > 0 or list_count > 0 or map_count > 0 or object_count > 0:
        return "O(N) (Dynamic Memory Allocation Detected)"
    else:
        return "O(1) (Constant - Primitive Variables Only)"

reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English

# Home Page
@app.route("/")
def index():
    """Serve the homepage (codelytics.html)."""
    return render_template("codelytics.html")

# Image Analysis Page
@app.route("/image")
def image_page():
    """Serve image.html."""
    return render_template("image.html")

# Text Readability Page
@app.route("/text")
def text_page():
    """Serve text.html."""
    return render_template("text.html")

# Analyze Code Endpoint (Restored)
@app.route("/analyze", methods=["POST"])
def analyze():
    """Process Java code and return a readability score."""
    data = request.get_json()
    java_code = data.get("code", "")

    numerical_score = predict_readability(java_code)
    llama_score = classify_readability_with_llama(java_code)
    final_score = compute_final_score(numerical_score, llama_score)

    # Bug Localization Analysis
    ml_prediction = predict_bug_status_ml(java_code)
    groq_prediction = classify_bug_with_groq(java_code)
    final_bug_status = compute_final_bug_score(ml_prediction, groq_prediction)

    # Run Complexity Analysis
    complexity_analysis = estimate_time_complexity(java_code)
    space_complexity = estimate_space_complexity(java_code)

    return jsonify({
        "score": final_score,
        "bug_status":final_bug_status,
        "complexity_analysis": complexity_analysis,
        "space_complexity": space_complexity
    })

def extract_text_from_image(image_path):
    """Extracts text from an image using Tesseract OCR."""
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def save_text_to_file(text, filename):
    """Saves extracted text to a file in the uploads folder."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Extracted text saved to {file_path}")

ANALYZE_ENDPOINT = "http://localhost:5000/analyze"

@app.route("/upload-image", methods=["POST"])
def upload_image():
    """Uploads an image, extracts Java code from it, and analyzes the code."""
    
    if "image" not in request.files:
        return jsonify({"message": "No file uploaded."}), 400

    image = request.files["image"]
    
    # Ensure the uploads folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Extract text from the image (OCR)
    extracted_text = extract_text_from_image(image_path)
    
    if not extracted_text:
        return jsonify({"message": "Failed to extract text from the image."}), 400

    # Save extracted text to a file
    text_filename = image.filename.rsplit(".", 1)[0] + ".txt"
    save_text_to_file(extracted_text, text_filename)

    # Run code analysis on extracted text
    numerical_score = predict_readability(extracted_text)
    llama_score = classify_readability_with_llama(extracted_text)
    final_score = compute_final_score(numerical_score, llama_score)

    # Bug Localization Analysis
    ml_prediction = predict_bug_status_ml(extracted_text)
    groq_prediction = classify_bug_with_groq(extracted_text)
    final_bug_status = compute_final_bug_score(ml_prediction, groq_prediction)

    # Complexity Analysis
    complexity_analysis = estimate_time_complexity(extracted_text)
    space_complexity = estimate_space_complexity(extracted_text)

    return jsonify({
        "message": "Image uploaded and processed successfully.",
        "extracted_code": extracted_text,
        "score": final_score,
        "bug_status": final_bug_status,
        "complexity_analysis": complexity_analysis,
        "space_complexity": space_complexity
    })




if __name__ == "__main__":
    app.run(debug=True)
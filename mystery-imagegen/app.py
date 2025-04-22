import os
import random
import requests
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from dotenv import load_dotenv
from image import MysteryBot

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize MysteryBot
mystery_bot = MysteryBot()

# Mystery prompts with anachronisms or impossible elements
MYSTERY_PROMPTS = [
    {
        "prompt": "A medieval knight holding a smartphone",
        "anachronism": "smartphone",
        "explanation": "Smartphones didn't exist in medieval times!"
    },
    {
        "prompt": "A dinosaur walking through a modern city",
        "anachronism": "dinosaur",
        "explanation": "Dinosaurs went extinct millions of years before modern cities existed!"
    },
    {
        "prompt": "A Victorian lady using a drone",
        "anachronism": "drone",
        "explanation": "Drones are modern technology that didn't exist in the Victorian era!"
    },
    {
        "prompt": "A pirate ship with a satellite dish",
        "anachronism": "satellite dish",
        "explanation": "Satellite dishes weren't invented during the age of pirates!"
    },
    {
        "prompt": "A caveman with a laptop",
        "anachronism": "laptop",
        "explanation": "Laptops are modern technology that didn't exist in prehistoric times!"
    },
    {
        "prompt": "A Roman soldier with a wristwatch",
        "anachronism": "wristwatch",
        "explanation": "Wristwatches weren't invented in ancient Rome!"
    },
    {
        "prompt": "A Studio Ghibli-style scene of a samurai with a laptop",
        "anachronism": "laptop",
        "explanation": "Laptops didn't exist in feudal Japan during the samurai era!"
    }
]

# Multiple choice options for each prompt
def generate_options(correct_answer):
    all_anachronisms = [item["anachronism"] for item in MYSTERY_PROMPTS]
    incorrect_options = [item for item in all_anachronisms if item != correct_answer]
    options = random.sample(incorrect_options, 2) + [correct_answer]
    random.shuffle(options)
    return options

# We're now using the MysteryBot class from image.py instead of this function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the analysis preference from the request
    include_analysis = request.form.get('include_analysis') == 'true'
    
    # Use MysteryBot to create a mystery
    mystery_result = mystery_bot.create_mystery(include_analysis=include_analysis)
    
    if not mystery_result["success"]:
        return jsonify({
            "success": False,
            "error": mystery_result["error"]
        })
    
    # Store the mystery in the session
    session["current_mystery"] = {
        "prompt": mystery_result["prompt"],
        "anachronism": mystery_result["anachronism"],
        "explanation": mystery_result["explanation"],
        "image_url": mystery_result["image_url"]
    }
    
    # If we have additional analysis, store it too
    if "object_detection" in mystery_result:
        session["current_mystery"]["object_detection"] = mystery_result["object_detection"]
    
    if "anachronism_details" in mystery_result:
        session["current_mystery"]["anachronism_details"] = mystery_result["anachronism_details"]
    
    # Store options in session
    session["options"] = mystery_result["options"]
    
    response_data = {
        "success": True,
        "image_url": mystery_result["image_url"],
        "options": mystery_result["options"]
    }
    
    # Add analysis to response if available
    if include_analysis:
        if "object_detection" in mystery_result:
            response_data["object_detection"] = mystery_result["object_detection"]
        if "anachronism_details" in mystery_result:
            response_data["anachronism_details"] = mystery_result["anachronism_details"]
    
    return jsonify(response_data)

@app.route('/check_answer', methods=['POST'])
def check_answer():
    user_answer = request.form.get('answer')
    
    if "current_mystery" not in session:
        return jsonify({
            "success": False,
            "error": "No active mystery found"
        })
    
    mystery = session["current_mystery"]
    correct = mystery_bot.check_answer(user_answer, mystery["anachronism"])
    
    response = {
        "success": True,
        "correct": correct,
        "explanation": mystery["explanation"] if correct else f"Incorrect! The anachronism was the {mystery['anachronism']}."
    }
    
    # Add detailed analysis if available
    if "anachronism_details" in mystery and correct:
        response["detailed_explanation"] = mystery["anachronism_details"]
    
    # Add object detection results if available
    if "object_detection" in mystery:
        response["object_detection"] = mystery["object_detection"]
    
    return jsonify(response)

@app.route('/download_image')
def download_image():
    if "current_mystery" not in session or "image_url" not in session["current_mystery"]:
        return redirect(url_for('index'))
    
    # Redirect to the image URL with download headers
    image_url = session["current_mystery"]["image_url"]
    return render_template('download.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)

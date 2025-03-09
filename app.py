from flask import Flask, render_template, request, flash, redirect, url_for, session,jsonify

import sqlite3
import random
import numpy as np
import csv
import pickle
import json
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import re
import joblib
from sklearn.ensemble import RandomForestClassifier


lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_mode2.h5")
filename="intents1.json"
intents = json.loads(open(filename, encoding="utf8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
app.secret_key = "123"

import sqlite3

DATABASE= 'database.db'

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

# Create users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    useremail TEXT NOT NULL,
    password TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS disease (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    useremail TEXT NOT NULL,
    predicted_disease TEXT NOT NULL
)
''')
cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            words TEXT NOT NULL
        )
    """)

conn.commit()
conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        useremail = request.form['useremail']
        password = request.form['password']
        #hashed_password = generate_password_hash(password, method='sha256')

        conn = sqlite3.connect(DATABASE)

        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE useremail = ?', (useremail,))
        user = cursor.fetchone()

        if user:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        cursor.execute('INSERT INTO users (username, useremail,password) VALUES (?, ?,?)', (username, useremail, password))
        conn.commit()
        conn.close()

        flash("Registration successful!", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        useremail = request.form['useremail']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE useremail = ? AND password = ? ', (useremail, password,))
        user = cursor.fetchone()
        print(user)

        if user:
            session ['useremail'] = useremail
            flash("Login successful!", "success")
            return render_template('index1.html')

        flash("Invalid username or password!", "danger")
        conn.close()
        return redirect(url_for('login'))

    return render_template('login.html')




@app.route('/')
def index():
    return render_template('register.html')





@app.route('/index1')
def index1():
    return render_template('index1.html')






@app.route('/logout')
def logout():
       session.clear()
       return redirect(url_for("index"))


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


# NLP Setup
lemmatizer = WordNetLemmatizer()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib  # To save and load the model

# Sample dataset (expand this with real data)
data = [
    (["fever", "headache", "muscle pain", "vomiting"], "Malaria"),
    (["fever", "fatigue", "sore throat", "cough"], "Viral Fever"),
    (["fever", "headache", "muscle pain", "fatigue", "vomiting", "rashes", "eye pain"], "Dengue Fever"),
    (["fever", "joint pain", "skin rashes", "eye redness"], "Dengue Fever"),
    (["fever", "nausea and vomiting", "severe headache", "fatigue"], "Viral Fever"),
    (["fever", "high fever", "abdominal pain", "bleeding from nose", "bleeding from gums"], "Dengue Fever"),
    (["fever", "stomach pain", "fatigue", "vomiting"], "Malaria"),
    (["fever", "muscle pain", "joint pain", "skin rashes"], "Dengue Fever"),
    (["fever", "cough", "chest pain", "difficulty breathing"], "Viral Fever"),
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["symptoms", "disease"])

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "mlb.pkl")

model = joblib.load("disease_model.pkl")
mlb = joblib.load("mlb.pkl")


disease_info = {
    "Dengue Fever": {
        "prevention": "Avoid mosquito bites, use repellents, and sleep under nets.",
        "severity": "Can lead to severe complications like Dengue Hemorrhagic Fever.",
        "doctor_check": "Visit a doctor if you have persistent fever, rashes, or bleeding.",
        "medication": "Paracetamol for fever, stay hydrated, and avoid aspirin.",
        "lifestyle": "Drink plenty of fluids and get enough rest."
    },

    "Malaria": {
        "prevention": "Use mosquito nets, insect repellents, and remove stagnant water.",
        "severity": "Can cause organ failure if untreated.",
        "doctor_check": "Get tested if experiencing fever and chills.",
        "medication": "Antimalarial drugs like chloroquine as prescribed.",
        "lifestyle": "Stay hydrated and take proper rest."
    },
    "Viral Fever": {
        "prevention": "Avoid crowded places, wash hands frequently, and stay hydrated.",
        "severity": "Mostly mild but can cause high fever and weakness.",
        "doctor_check": "Consult a doctor if fever persists beyond 4-5 days.",
        "medication": "Paracetamol for fever, rest, and fluids.",
        "lifestyle": "Eat nutritious food, stay warm, and take adequate rest."
    }
   
}

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.json.get("symptoms", [])
    print("Received Symptoms:", selected_symptoms)
    user_email = session.get('useremail')  # Ensure session contains useremail

    # Tamil-to-English translation dictionary
    tamil_to_english = {
        "காய்ச்சல்": "fever",
        "தலைவலி": "headache",
        "மலச்சிக்கல்": "constipation",
        "வாந்தி": "vomiting",
        "வயிற்று வலி": "stomach pain",
        "மாசில் வலி": "muscle pain",
        "சோர்வு": "fatigue",
        "சினப்பு தோல்": "rashes",
        "மூக்கில் இருந்து இரத்தசேதம்": "bleeding from nose",
        "பற்களில் இருந்து இரத்தசேதம்": "bleeding from gums",
        "மூட்டுவலி": "joint pain",
        "கண்வலி": "eye pain",
        "உயர் காய்ச்சல்": "high fever",
        "தோல் சினப்பு": "skin rashes",
        "குடல் வலி": "abdominal pain",
        "மார்பு வலி": "chest pain",
        "கண் சிவப்பு": "eye redness",
        "இருமல்": "cough",
        "வயிற்றுப் போக்கு மற்றும் வாந்தி": "nausea and vomiting",
        "கடுமையான தலைவலி": "severe headache",
        "தொண்டை வலி": "sore throat",
        "மூச்சுத் திணறல்": "difficulty breathing"
    }

    english_to_tamil = {v: k for k, v in tamil_to_english.items()}  # Reverse dictionary

    # Tamil disease and suggestion mapping
    tamil_disease_info = {
        "Dengue Fever": {
            "name": "டெங்கு காய்ச்சல்",
            "prevention": "பூச்சிக்கொல்லிகளை பயன்படுத்தவும், கொசு கடியை தவிர்க்கவும்.",
            "severity": "கடுமையான பாதிப்புகளைக் கொண்டிருக்கலாம்.",
            "doctor_check": "தொடர் காய்ச்சல் இருந்தால் மருத்துவரை அணுகவும்.",
            "medication": "பிரசிதமோல் மற்றும் அதிக திரவங்களை பருகவும்.",
            "lifestyle": "பெரும் அளவில் நீர் பருகவும் மற்றும் ஓய்வு எடுக்கவும்."
        },
        "Malaria": {
            "name": "மலேரியா",
            "prevention": "கொசு கடியை தவிர்க்கவும், நெட்ஸ் பயன்படுத்தவும்.",
            "severity": "சிகிச்சை பெறாதால் உடலுக்கு தீங்கு விளைவிக்கும்.",
            "doctor_check": "காய்ச்சல் இருந்தால் பரிசோதிக்க வேண்டும்.",
            "medication": "கொடுக்கப்பட்ட மருந்துகளை மட்டும் எடுத்துக்கொள்ளவும்.",
            "lifestyle": "உடலுக்கு போதிய ஓய்வு மற்றும் நீர் பருகுதல் அவசியம்."
        },
        "Viral Fever": {
            "name": "வைரஸ் காய்ச்சல்",
            "prevention": "கைகழுவுதல், பொது இடங்களை தவிர்த்தல்.",
            "severity": "வலுவான காய்ச்சல் மற்றும் சோர்வை ஏற்படுத்தலாம்.",
            "doctor_check": "காய்ச்சல் நீண்டால் மருத்துவரை அணுகவும்.",
            "medication": "பிரசிதமோல், ஓய்வு மற்றும் திரவங்களை பருகுதல்.",
            "lifestyle": "நல்ல உணவு, சூடான நீர் பருகுதல், ஓய்வெடுத்தல்."
        }
    }

    # Function to clean the symptoms (remove unwanted text and newlines)
    def clean_symptom(symptom):
        return symptom.split("\n")[0].strip()  # Take only the first part before newline

    # Clean symptoms
    cleaned_symptoms = [clean_symptom(symptom) for symptom in selected_symptoms]

    # Detect if symptoms are in Tamil or English
    is_tamil = any(symptom in tamil_to_english for symptom in cleaned_symptoms)

    if is_tamil:
        translated_symptoms = [tamil_to_english.get(symptom, symptom) for symptom in cleaned_symptoms]
    else:
        translated_symptoms = cleaned_symptoms  # Symptoms are already in English

    print("Cleaned Symptoms:", cleaned_symptoms)
    print("Translated Symptoms:", translated_symptoms)

    # If fever is not present, return a message
    if "fever" not in translated_symptoms:
        return jsonify({"predicted_disease": "No fever detected", "suggestions": {}})

    # Transform input symptoms into the format used in training
    input_data = mlb.transform([translated_symptoms])

    # Predict disease
    predicted_disease = model.predict(input_data)[0]

    # Convert prediction and suggestions to Tamil if the input was Tamil
    if is_tamil:
        predicted_disease_tamil = tamil_disease_info.get(predicted_disease, {}).get("name", predicted_disease)
        suggestions = tamil_disease_info.get(predicted_disease, {})
    else:
        predicted_disease_tamil = predicted_disease  # Keep it in English
        suggestions = disease_info.get(predicted_disease, {})

    # Insert the predicted disease into the database linked to the user
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Check if the email exists in the users table
    cursor.execute('SELECT * FROM users WHERE useremail = ?', (user_email,))
    user = cursor.fetchone()

    if user:
        # Insert predicted disease for the user
        cursor.execute('''
        INSERT INTO disease (username, useremail, predicted_disease)
        VALUES (?, ?, ?)
        ''', (user[1], user[2], predicted_disease))  # Assuming user[1] is username and user[2] is useremail

        conn.commit()
    
    conn.close()

    # Return the prediction and suggestions in the correct language
    return jsonify({
        "predicted_disease": predicted_disease_tamil,
        "suggestions": suggestions
    })



@app.route('/result')
def result():
    disease = request.args.get('disease', 'No prediction available')
    return render_template("result.html", disease=disease)

# User response storage
user_data = []
current_step = 0  # To track which question to ask next

def get_intent_response(msg):
    """Check if user input matches any known intents."""
    msg = msg.lower().strip()
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if re.search(pattern, msg):
                return random.choice(intent["responses"])  # Return a random response from the list
    return None

def process_message(msg):
    """பயனர் உள்ளீட்டை செயலாக்கி அடுத்த பதிலை தீர்மானிக்கிறது."""
    global user_data, current_step

    msg = msg.lower().strip()

    intent_response = get_intent_response(msg)
    if intent_response:
        return intent_response
    
    return "மன்னிக்கவும், நான் புரிந்துகொள்ளவில்லை. தயவுசெய்து வழிமுறைகளை பின்பற்றவும்."



def store_detected_words(email, words):
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO detected_words (email, words) VALUES (?, ?)", (email, ", ".join(words)))
    conn.commit()
    conn.close()


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    user_email = session["useremail"]  # Get logged-in user's email

    result = process_message(msg)  # Get chatbot response

    # Keywords to check
    keywords = ["மலேரியா", "டெங்கு", "வைரல்"]
    detected_words = [word for word in keywords if word in result]

    # Store in database if keywords are found
    if detected_words and user_email:
        store_detected_words(user_email, detected_words)

    return jsonify({"response": result})

def bow(sentence, words, show_details=True):
    """Converts a sentence into a bag-of-words representation."""
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize words

    bag = [0] * len(words)  # Initialize bag-of-words with zeroes
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # Mark the word as present
                if show_details:
                    print(f"Found in bag: {w}")

    return np.array(bag)


def predict_class(sentence, model):
    """Predicts intent using the chatbot model."""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list


def getResponse(ints, intents_json):
    """Handles chatbot responses based on predicted intent."""
    if not ints:
        return "Sorry, I don't understand that."

    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])

    return "Sorry, I don't understand that."




r = sr.Recognizer()
mic = sr.Microphone()

import speech_recognition as sr

def speak_tamil():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as audio_file:
        print("Speak Now...")
        r.adjust_for_ambient_noise(audio_file)
        try:
            audio = r.listen(audio_file, timeout=5)  # Add a timeout to avoid hanging
            print("Audio received. Converting to text...")
            text = r.recognize_google(audio, language='ta-IN')
            text = text.lower()
            print("User Input (Tamil):", text)
            return text
        except sr.UnknownValueError:
            print("Speech not clear")
            return "Speech not clear, please try again."
        except sr.RequestError:
            print("Recognition service unavailable")
            return "Speech recognition service unavailable."
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "An error occurred during speech recognition."

# Run the function
speak_tamil()

# Function to convert Tamil text to speech and save as an audio file
def generate_tamil_speech(text):
    audio_dir = "static"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    for filename in os.listdir(audio_dir):
        if filename.startswith("response_") and filename.endswith(".mp3"):
            os.remove(os.path.join(audio_dir, filename))

    response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
    response_audio_path = os.path.join(audio_dir, response_audio_filename)

    try:
        tts = gTTS(text=text, lang='ta')
        tts.save(response_audio_path)
        return response_audio_filename
    except Exception as e:
        return None

@app.route('/speaktamil', methods=['GET', 'POST'])
def speaktamil():
    print("Start Tamil Speech Processing")

    # Step 1: Capture Tamil speech
    speech_tamil = speak_tamil()
    
    # If speech is not recognized or service is unavailable, return an error message
    if "Speech not clear" in speech_tamil or "service unavailable" in speech_tamil:
        return render_template('chatbot.html', error=speech_tamil)

    # Step 2: Process Tamil message directly in chatbot logic
    result_tamil = process_message(speech_tamil)  # Direct Tamil processing
    print("Response in Tamil:", result_tamil)
    keywords = ["மலேரியா", "டெங்கு", "வைரல்"]
    user_email = session['useremail']
    detected_words = [word for word in keywords if word in result_tamil]

    # Store in database if keywords are found
    if detected_words and user_email:
        user_email = session['useremail']
        store_detected_words(user_email, detected_words)

    # Step 3: Convert Tamil response to speech
    response_audio_filename = generate_tamil_speech(result_tamil)
    
    if not response_audio_filename:
        return render_template('chatbot.html', error="Error generating Tamil speech audio")

    # Render chatbot.html with the processed data
    return render_template(
        'chatbot.html',
        audio_file=response_audio_filename,
        speech=speech_tamil,
        result=result_tamil
    )

if __name__ == '__main__':
    app.run(port=2000)

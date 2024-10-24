from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import joblib
import sqlite3
from tensorflow.keras import backend as K
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and label encoder
model = load_model('model/alzheimer_model.h5')
le = joblib.load('model/label_encoder.pkl')

# Class mapping
class_mapping = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non-Demented",
    3: "Very Mild Dementia"
}

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    return image

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('predict'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Email already exists')
    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = preprocess_image(img)
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            predicted_class = np.argmax(preds, axis=1)
            predicted_index = predicted_class[0]
            prediction = class_mapping.get(predicted_index, "Unknown Class")

            K.clear_session()

    return render_template('index.html', prediction=prediction)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

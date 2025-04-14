import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("densenet_model/model_v1.keras")  

# Class label mapping
class_names = ["Healthy", "Scab", "Rust", "Multiple disease"]

def preprocess_image(image):
    image = image.resize((512, 512))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    image = Image.open(file).convert('RGB')
    input_data = preprocess_image(image)

    prediction = model.predict(input_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    predicted_label = class_names[predicted_class]

    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)

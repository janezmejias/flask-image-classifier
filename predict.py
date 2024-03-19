import os

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = 'fit_categorical_model.h5'
model = load_model(model_path)
print("Model loaded OK." if model else "Problem loading model.")

classes = {0: 'fb', 1: 'idea', 2: 'yt'}


@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No imagen available'}), 400

        file = request.files['image']
        filename = secure_filename(file.filename)

        temp_dir = os.path.join('../temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # made prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        probability = float(prediction[0][predicted_class_index])

        os.remove(filepath)

        return jsonify({'class': predicted_class, 'probability': probability})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Error processing the image'}), 500


if __name__ == '__main__':
    app.run(debug=False)

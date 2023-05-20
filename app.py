import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model_path = './test.h5'
model = load_model(model_path)
image_size = (64, 64)  # Set the desired image size

app = Flask(__name__,template_folder='templateFiles', static_folder='staticFiles')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload',methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/result',methods=['GET','POST'])
def result():
        if request.method == 'POST':
            # Save the uploaded image
            img = request.files['images']
            img_path = 'uploaded_image.jpg'
            img.save(img_path)

            # Preprocess the image
            img = image.load_img(img_path, target_size=None)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            # Normalize pixel values
            img = img / 300.0

            # Make predictions
            prediction = model.predict(img)
            if prediction[0][0] < 0.01:
                result = 'Disease'
            else:
                result = 'Healthy'

            # Remove the uploaded image
            os.remove(img_path)

            return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

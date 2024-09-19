from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import rasterio
from PIL import Image
import io
import base64
import os

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set maximum upload size to 100MB

# Define the FixedDropout layer
@tf.keras.utils.register_keras_serializable()
class FixedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        return tf.keras.backend.in_train_phase_v2(
            lambda: tf.nn.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape, seed=self.seed),
            lambda: inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        config.update({
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        })
        return config

# Import swish activation
from tensorflow.keras.activations import swish

# Load your pre-trained model with custom_objects for swish and FixedDropout
model = tf.keras.models.load_model('water_segmentation_model_pre_trained.keras', custom_objects={'swish': swish, 'FixedDropout': FixedDropout})

# Function to normalize images
def normalize_images(images):
    images_norm = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[-1]):
        min_val = np.min(images[..., i])
        max_val = np.max(images[..., i])
        images_norm[..., i] = (images[..., i] - min_val) / (max_val - min_val)
    return images_norm

# Function to process and predict
def process_image(file):
    try:
        with rasterio.open(file) as src:
            image = src.read()  # Reading the 12-channel image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    image = np.moveaxis(image, 0, -1)  # Move the channels to the last dimension
    image = normalize_images(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the segmentation
    try:
        prediction = model.predict(image)[0]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

    # Convert the prediction to an image format
    output_image = (prediction[..., 0] * 255).astype(np.uint8)
    
    output_img = Image.fromarray(output_image)
    img_io = io.BytesIO()
    output_img.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io

# Function to convert image to base64
def encode_image_to_base64(img_io):
    img_str = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))  # Redirect to index if no file is uploaded

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))  # Redirect if no file is selected

    if file:
        # Process the image and get output
        output_img = process_image(file)
        if output_img is None:
            flash("Error processing image")
            return redirect(url_for('index'))

        # Convert input TIFF to a displayable format
        try:
            with rasterio.open(file) as src:
                input_image = src.read()
            input_image = np.moveaxis(input_image, 0, -1)
            input_image = normalize_images(input_image)
            input_image_pil = Image.fromarray((input_image[..., 0] * 255).astype(np.uint8))

            # Convert input and output images to base64
            img_io_input = io.BytesIO()
            input_image_pil.save(img_io_input, 'PNG')
            img_io_input.seek(0)

            input_image_base64 = encode_image_to_base64(img_io_input)
            output_image_base64 = encode_image_to_base64(output_img)

            # Render the result page
            return render_template('result.html', input_image=input_image_base64, output_image=output_image_base64)
        except Exception as e:
            print(f"Error rendering result: {e}")
            flash("Error rendering result")
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

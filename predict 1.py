import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Path to input image')
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model('models/best_model.h5')
img_path = args.image

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

preds = model.predict(x)
class_idx = np.argmax(preds[0])
print(f"Predicted class: {class_idx}")

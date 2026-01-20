import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

model_path = "models/plant_disease_recog_model_pwp.keras"
image_path = "examples/plant.jpg"
json_path = "plant_disease.json"
output_file = "prediction_results.txt"

def predict_with_size(model, img, size, class_names):
    img_resized = img.convert("RGB").resize(size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    idx = np.argmax(probabilities)
    conf = probabilities[idx] * 100
    
    predicted_class = class_names[idx]["name"]
    return predicted_class, conf, probabilities

with open(output_file, "w") as f:
    try:
        model = tf.keras.models.load_model(model_path)
        with open(json_path, "r") as jf:
            class_names = json.load(jf)
            
        img = Image.open(image_path)
        
        f.write("--- Testing 160x160 ---\n")
        pred_160, conf_160, probs_160 = predict_with_size(model, img, (160, 160), class_names)
        f.write(f"Prediction: {pred_160}\nConfidence: {conf_160:.2f}%\n")
        f.write(f"Top 3 indices: {np.argsort(probs_160)[-3:][::-1]}\n")
        f.write(f"Top 3 probs: {np.sort(probs_160)[-3:][::-1]}\n\n")

        f.write("--- Testing 224x224 ---\n")
        pred_224, conf_224, probs_224 = predict_with_size(model, img, (224, 224), class_names)
        f.write(f"Prediction: {pred_224}\nConfidence: {conf_224:.2f}%\n")
        f.write(f"Top 3 indices: {np.argsort(probs_224)[-3:][::-1]}\n")
        f.write(f"Top 3 probs: {np.sort(probs_224)[-3:][::-1]}\n")
        
    except Exception as e:
        f.write(f"Error: {str(e)}\n")

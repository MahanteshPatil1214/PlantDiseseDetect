import sys
import tensorflow as tf
import os

model_path = "models/plant_disease_recog_model_pwp.keras"
output_file = "model_info.txt"

with open(output_file, "w") as f:
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            f.write("Model loaded successfully.\n")
            f.write(f"Input Shape: {model.input_shape}\n")
            
            # Capture summary
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_summary = "\n".join(stringlist)
            f.write(short_summary)
            
        except Exception as e:
            f.write(f"Error loading model: {e}\n")
    else:
        f.write(f"Model file not found at {model_path}\n")


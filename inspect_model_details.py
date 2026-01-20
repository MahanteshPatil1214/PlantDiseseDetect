import tensorflow as tf
import os

model_path = "models/plant_disease_recog_model_pwp.keras"
output_file = "model_details_v2.txt"

with open(output_file, "w") as f:
    f.write(f"Inspecting model: {model_path}\n")

    try:
        model = tf.keras.models.load_model(model_path)
        f.write("\n--- Model Input Shape ---\n")
        f.write(str(model.input_shape) + "\n")
        
        f.write("\n--- First Layer Config ---\n")
        try:
            config = model.layers[0].get_config()
            f.write(str(config) + "\n")
        except:
            f.write("Could not get first layer config\n")

        f.write("\n--- First 10 Layers ---\n")
        for i, layer in enumerate(model.layers[:10]):
            f.write(f"{i}: {layer.name} ({layer.__class__.__name__})\n")
    
    except Exception as e:
        f.write(f"Error: {e}\n")

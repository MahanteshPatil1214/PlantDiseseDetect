import streamlit as st
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import io # To handle the file buffer from Streamlit uploader

# --- CONFIGURATION AND DATA LOADING ---

# Changed layout="centered" to layout="wide" for a website look
st.set_page_config(
    page_title="Agri-Sense: Plant Disease Diagnostic",
    layout="wide", 
    initial_sidebar_state="auto"
)

# Use Streamlit caching to load the model only once
# @st.cache_resource is the modern way to cache models/resources
@st.cache_resource
def load_model():
    """Loads the Keras model once."""
    try:
        # NOTE: Ensure the file path is correct relative to where you run the script.
        model_path = "models/plant_disease_recog_model_pwp.keras"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop the script if the model can't load

# Use Streamlit caching to load JSON data only once
@st.cache_data
def load_disease_data():
    """Loads the plant disease metadata/labels."""
    try:
        with open("plant_disease.json", 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Error: 'plant_disease.json' not found. Please ensure it's in the same directory.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Error: Could not decode 'plant_disease.json'. Check its format.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading JSON: {e}")
        st.stop()

# Load resources
model = load_model()
plant_disease = load_disease_data()

# --- PREPROCESSING AND PREDICTION FUNCTIONS ---

def extract_features(image_buffer, target_size=(160, 160)):
    """
    Loads an image from a buffer, resizes it, and converts it to a NumPy array for model input.
    """
    try:
        # Open the image using PIL (Pillow)
        image = Image.open(image_buffer)
        
        # Resize the image
        image = image.resize(target_size)
        
        # Convert to NumPy array
        feature = tf.keras.utils.img_to_array(image)
        
        # Add batch dimension (1, H, W, C)
        feature = np.array([feature])
        return feature
    except Exception as e:
        st.error(f"Error during image processing: {e}")
        return None

def model_predict(image_buffer):
    """
    Predicts the disease label from an image buffer.
    """
    # 1. Extract features
    img_features = extract_features(image_buffer)
    
    if img_features is None:
        return None, None # Return early if feature extraction failed

    # 2. Predict using the model
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_features)
    
    # 3. Get the predicted index and label
    predicted_index = prediction.argmax()
    
    # Check if the index is valid
    if 0 <= predicted_index < len(plant_disease):
        prediction_label = plant_disease[predicted_index]
        confidence = prediction[0][predicted_index] * 100 # Calculate confidence percentage
        return prediction_label, confidence
    else:
        st.error("Prediction index out of bounds. Check your model output and JSON labels.")
        return None, None


# --- STREAMLIT UI LAYOUT ---

# 1. CUSTOM NAVIGATION BAR (Website Look)
st.markdown(
    """
    <style>
    /* Global CSS styles are at the bottom for better organization */
    .header {
        background-color: #38761D; /* Deep Green for Header */
        color: black;
        padding: 15px 25px; /* Increased padding */
        font-size: 26px;
        font-weight: 800;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15); /* Stronger shadow */
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-content {
        padding-top: 80px; /* Adjust content down for new header size */
        padding-left: 20px;
        padding-right: 20px;
    }
    /* Hide the default Streamlit main header space */
    .css-18e3th9 {
        padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="header">🌱 Agri-Sense: Plant Disease Diagnostic</div>',
    unsafe_allow_html=True
)

# Main content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) # Spacer for better vertical alignment

# Introductory Text
st.header("Upload a Leaf for Instant Diagnosis")
st.markdown(
    "Our AI-powered tool analyzes leaf images to identify common diseases and provide immediate treatment recommendations. Simple, fast, and effective farm intelligence."
)
st.markdown("---")

# Layout columns for the main interaction area
col_upload, col_result = st.columns(2)

# --- UPLOAD COLUMN ---
with col_upload:
    st.subheader("Step 1: Upload Image")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, or PNG) for disease detection.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="file_uploader"
    )
    
    # Placeholder for image preview
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Uploaded Leaf Preview")
        image_buffer = io.BytesIO(uploaded_file.read())
        st.image(image_buffer, caption=uploaded_file.name, use_column_width=True)

# --- RESULT/PREDICTION COLUMN ---
with col_result:
    st.subheader("Step 2: Diagnosis & Treatment")
    
    if uploaded_file is None:
        st.info("Upload an image in the left panel to begin the analysis.")
    else:
        # Center the diagnosis button
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button('🔎 Diagnose Disease', key='diagnose_button', help="Click to run the model prediction."):
            
            # Reset buffer pointer to the beginning before prediction
            image_buffer.seek(0) 
            
            # Get prediction and confidence
            prediction_label, confidence = model_predict(image_buffer)
            
            if prediction_label:
                st.success("✅ **Diagnosis Complete**")
                
                # --- Display Key Metrics in a Card (Improved Styling) ---
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="result-card">
                    <h4>Result Summary</h4>
                    <div class="metric-row">
                        <div class="metric-item">
                            <p class="metric-label">Predicted Disease</p>
                            <p class="metric-value-text">""" + prediction_label.get('name', 'Unknown') + """</p>
                        </div>
                        <div class="metric-item">
                            <p class="metric-label">Confidence Score</p>
                            <p class="metric-value-conf">""" + f"{confidence:.2f}%" + """</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                
                # --- Display detailed information in Thematic Expanders ---
                
                st.markdown("<h4>Detailed Information</h4>", unsafe_allow_html=True)

                # Cause Expander (Thematic red/yellow for warning/disease)
                with st.expander("🦠 VIEW CAUSE AND DESCRIPTION", expanded=True):
                    st.markdown("### Cause of Infection")
                    st.markdown(
                        f"<p style='padding: 15px; border-left: 6px solid #C55A11; background-color: #fcefe3; border-radius: 8px; font-style: italic;'><b>{prediction_label.get('cause', 'Cause information not available.')}</b></p>",
                        unsafe_allow_html=True
                    )
                
                # Cure Expander (Thematic deep green for solution/treatment)
                with st.expander("🌿 RECOMMENDED CURE AND TREATMENT", expanded=True):
                    st.markdown("### Recommended Treatment")
                    st.markdown(
                        f"<p style='padding: 15px; border-left: 6px solid #38761D; background-color: #e8f5e9; border-radius: 8px;'><b>{prediction_label.get('cure', 'Cure information not available.')}</b></p>",
                        unsafe_allow_html=True
                    )

                st.warning(
                   "Please consult a qualified agronomist for critical or widespread plant diseases."
                )
                
            else:
                st.error("Could not complete the diagnosis. Please try another image.")
        
        st.markdown("</div>", unsafe_allow_html=True) # Close center div


st.markdown('</div>', unsafe_allow_html=True) # Close main-content container

st.markdown("---", unsafe_allow_html=True) # Separator before footer

# 3. CUSTOM FOOTER
st.markdown(
    """
    <div class="footer">
        <p>Agri-Sense Diagnostic Tool | Built with Streamlit & TensorFlow | &copy; 2025 Farm Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- CUSTOM CSS FOR AGRICULTURE THEME ---
st.markdown(
    """
    <style>
    /* Main App Background - Slightly warmer off-white */
    .stApp {
        background-color: #fcfcfc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Custom Button Styling (Thematic Green) */
    .stButton>button {
        background-color: #38761D; /* Deep, natural green */
        color: white;
        font-weight: 700;
        border-radius: 10px; 
        padding: 12px 30px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25); /* More pronounced shadow */
        transition: background-color 0.3s ease;
        width: 100%; 
        max-width: 350px;
        margin-top: 15px;
    }
    .stButton>button:hover {
        background-color: #6AA84F; 
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* Lift on hover */
        color: white;
    }
    
    /* Style for Headers and Text - Earthy Tones */
    h1, h2, h3 {
        color: #C55A11; /* Earthy Brown/Terracotta for contrast */
    }
    .header {
        font-family: 'Arial Black', sans-serif; /* Stronger font for header */
    }
    h4 {
        color: #38761D; /* Green for sub-card titles */
        margin-top: 10px;
        margin-bottom: 10px;
        border-bottom: 2px solid #DCDCDC;
        padding-bottom: 5px;
    }
    
    /* Result Card Styling (Key visual element) */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15); /* Strong shadow for pop */
        margin-bottom: 25px;
        border: 1px solid #E0E0E0;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin-top: 15px;
    }
    .metric-item {
        flex: 1;
        padding: 15px;
        border-radius: 10px;
        background-color: #f5fff5; /* Very light green background */
        text-align: center;
        border: 1px solid #D9EAD3;
    }
    .metric-label {
        font-size: 15px;
        color: #777;
        margin: 0;
    }
    .metric-value-text {
        font-size: 24px;
        font-weight: 800;
        color: #C55A11; /* Earth tone for the disease name */
        margin: 5px 0 0 0;
    }
    .metric-value-conf {
        font-size: 24px;
        font-weight: 800;
        color: #28a745;
        margin: 5px 0 0 0;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 20px;
        background-color: #D9EAD3; /* Light green footer background */
        color: #38761D;
        font-size: 14px;
        border-top: 3px solid #6AA84F;
    }
    
    /* Input field styling */
    .stFileUploader {
        border: 2px dashed #B6D7A8; /* Dashed border for file drop */
        padding: 20px;
        border-radius: 10px;
        background-color: #f5f5f5;
    }
    
    </style>
    """, 
    unsafe_allow_html=True
)
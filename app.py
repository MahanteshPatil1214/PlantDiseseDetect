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
        # Placeholder path, assuming the user has the model file
        model_path = "models/plant_disease_recog_model_pwp.keras" 
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # In a real app, you might stop the script here
        # st.stop() 
        # For demonstration, returning a dummy model if loading fails
        st.warning("Using a dummy model for display purposes. Please ensure your model file is accessible.")
        # Dummy model structure for successful UI rendering
        class DummyModel:
             def predict(self, features):
                 # Dummy prediction returning index 0 with high confidence
                 return np.array([[0.9, 0.05, 0.05]])
        return DummyModel()

# Use Streamlit caching to load JSON data only once
@st.cache_data
def load_disease_data():
    """Loads the plant disease metadata/labels."""
    # Dummy data structure to ensure the app runs even without the file
    dummy_data = [
        {"name": "Tomato Bacterial Spot", "cause": "Caused by the bacteria Xanthomonas spp., forming dark, greasy spots.", "cure": "Apply copper-based bactericides and ensure good air circulation."},
        {"name": "Corn Common Rust", "cause": "Caused by the fungus Puccinia sorghi, appearing as reddish-brown pustules.", "cure": "Use rust-resistant varieties and foliar fungicides containing triazole."},
        {"name": "Potato Healthy", "cause": "No disease detected.", "cure": "Maintain optimal irrigation, fertilization, and general care."}
    ]
    
    try:
        with open("plant_disease.json", 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.warning("Warning: 'plant_disease.json' not found. Using dummy data for display.")
        return dummy_data
    except json.JSONDecodeError:
        st.error("Error: Could not decode 'plant_disease.json'. Check its format.")
        return dummy_data
    except Exception as e:
        st.error(f"An unexpected error occurred while loading JSON: {e}")
        return dummy_data

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
        
        # Ensure image is in RGB format before resizing
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize the image
        image = image.resize(target_size)
        
        # Convert to NumPy array
        # Note: Keras utility handles conversion better than raw np.array(image)
        feature = tf.keras.utils.img_to_array(image) 
        
        # Add batch dimension (1, H, W, C) and normalize (if necessary for the model)
        # Assuming the model expects values in the range [0, 255] or handles normalization internally
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
        # Explicitly cast to float32 before prediction if needed, though Keras often handles this
        prediction = model.predict(img_features.astype(np.float32)) 
    
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


# --- STREAMLIT UI LAYOUT (ENHANCED) ---

# 1. CUSTOM NAVIGATION BAR (Website Look)
# This CSS is now integrated into the final CSS block for cleanliness.

st.markdown(
    f'<div class="header">🌱 **Agri-Sense:** Plant Disease Diagnostic</div>',
    unsafe_allow_html=True
)

# Main content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) # Spacer for better vertical alignment

# Introductory Text
st.header("Upload a Leaf for Instant Diagnosis")
st.markdown(
    "Our **AI-powered tool** analyzes leaf images to identify common diseases and provide immediate treatment recommendations. Simple, fast, and effective farm intelligence."
)
st.markdown("---")

# Layout columns for the main interaction area
col_upload, col_result = st.columns(2)

# --- UPLOAD COLUMN ---
with col_upload:
    st.subheader("1. Upload Image")
    
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
        # Read the file once and create a buffer
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        image_buffer = io.BytesIO(image_bytes)
        
        st.image(image_buffer, caption=uploaded_file.name, use_column_width=True)
        # Store the buffer in session state so it persists
        st.session_state['image_buffer'] = image_buffer 

# --- RESULT/PREDICTION COLUMN ---
with col_result:
    st.subheader("2. Diagnosis & Treatment")
    
    # Retrieve buffer from session state if available
    image_buffer_from_state = st.session_state.get('image_buffer')

    if uploaded_file is None:
        st.info("Upload an image in the left panel to begin the analysis.")
    else:
        # Center the diagnosis button
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        # Check if the button was clicked
        if st.button('🔎 Diagnose Disease', key='diagnose_button', help="Click to run the model prediction."):
            
            if image_buffer_from_state:
                # Reset buffer pointer to the beginning before prediction
                image_buffer_from_state.seek(0) 
                
                # Get prediction and confidence
                prediction_label, confidence = model_predict(image_buffer_from_state)
                
                if prediction_label:
                    st.success("✅ **Diagnosis Complete**")
                    
                    # --- Display Key Metrics in a Card (Improved Styling) ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Result Summary</h4>
                        <div class="metric-row">
                            <div class="metric-item">
                                <p class="metric-label">Predicted Disease</p>
                                <p class="metric-value-text">**{prediction_label.get('name', 'Unknown')}**</p>
                            </div>
                            <div class="metric-item">
                                <p class="metric-label">Confidence Score</p>
                                <p class="metric-value-conf">{confidence:.2f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    
                    # --- Display detailed information in Thematic Expanders ---
                    
                    st.markdown("<h4>Detailed Information and Recommendation</h4>", unsafe_allow_html=True)

                    # Cause Expander (Thematic red/yellow for warning/disease)
                    with st.expander("🦠 VIEW CAUSE AND DESCRIPTION", expanded=True):
                        st.markdown("##### Cause of Infection")
                        st.markdown(
                            f"<p style='padding: 15px; border-left: 6px solid #C55A11; background-color: #fcefe3; border-radius: 8px; font-style: italic;'>**{prediction_label.get('cause', 'Cause information not available.')}**</p>",
                            unsafe_allow_html=True
                        )
                    
                    # Cure Expander (Thematic deep green for solution/treatment)
                    with st.expander("🌿 RECOMMENDED CURE AND TREATMENT", expanded=True):
                        st.markdown("##### Recommended Treatment")
                        st.markdown(
                            f"<p style='padding: 15px; border-left: 6px solid #38761D; background-color: #e8f5e9; border-radius: 8px;'>**{prediction_label.get('cure', 'Cure information not available.')}**</p>",
                            unsafe_allow_html=True
                        )

                    st.warning(
                        "⚠️ **Disclaimer:** This is an AI-generated diagnosis. Please consult a qualified agronomist for critical or widespread plant diseases."
                    )
                    
                else:
                    st.error("Could not complete the diagnosis. Please try another image.")
            else:
                st.error("Error: Image buffer not found. Please re-upload the image.")
        
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


# --- CUSTOM CSS FOR AGRICULTURE THEME (Unified and Enhanced) ---
st.markdown(
    """
    <style>
    /* Global Streamlit Overrides */
    .stApp {
        background-color: #FAFAF5; /* Lighter, warmer earth tone */
        font-family: "Trebuchet MS", Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Bar Styling */
    .header {
        background-color: #38761D; /* Deep Green for Header */
        color: white;
        padding: 15px 25px; 
        font-size: 26px;
        font-weight: 800;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15); 
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
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

    
    /* Custom Button Styling (Thematic Green - made it bolder) */
    .stButton>button {
        background-color: #38761D; /* Deep, natural green */
        color: white;
        font-weight: 800; /* Bolder text */
        border-radius: 10px; 
        padding: 15px 35px; /* Bigger button */
        border: 2px solid #285A18; /* Subtle dark border */
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3); /* Stronger shadow for 3D feel */
        transition: all 0.2s ease;
        width: 100%; 
        max-width: 350px;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #6AA84F; 
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2); /* Flattern on hover */
        transform: translateY(2px); /* Slight press effect */
        color: white;
    }
    
    /* Style for Headers and Text - Earthy Tones */
    h1, .st-emotion-cache-1wa9n39 h2, .st-emotion-cache-1wa9n39 h3, .st-emotion-cache-1wa9n39 h4 {
        color: #C55A11; /* Earthy Brown/Terracotta for contrast */
        border-bottom: 2px solid #D9EAD3; /* Light green separator under header */
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-weight: 700; 
    }
    h4 {
        color: #38761D !important; 
        margin-top: 10px !important;
        margin-bottom: 10px !important;
        border-bottom: 2px solid #DCDCDC !important;
        padding-bottom: 5px !important;
    }
    
    /* Result Card Styling (Key visual element - more emphasis) */
    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2); /* Deep shadow */
        margin-bottom: 30px;
        border: 3px solid #6AA84F; /* Stronger green outline */
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px); /* Lift card on hover */
    }

    .metric-row {
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin-top: 20px;
    }
    .metric-item {
        flex: 1;
        padding: 20px;
        border-radius: 12px;
        background-color: #f5fff5; 
        text-align: center;
        border: 1px solid #D9EAD3;
    }
    .metric-label {
        font-size: 16px;
        color: #666;
        margin: 0;
    }
    .metric-value-text {
        font-size: 26px; /* Larger text */
        font-weight: 800;
        color: #C55A11; 
        margin: 8px 0 0 0;
    }
    .metric-value-conf {
        font-size: 26px; 
        font-weight: 800;
        color: #28a745;
        margin: 8px 0 0 0;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background-color: #D9EAD3; 
        color: #38761D;
        font-size: 15px;
        border-top: 4px solid #6AA84F; 
    }
    
    /* Input field styling (File Uploader) */
    .stFileUploader {
        border: 3px dashed #B6D7A8; /* Thicker dashed border */
        padding: 30px;
        border-radius: 15px;
        background-color: #f9f9f9;
        transition: background-color 0.3s ease;
    }
    .stFileUploader:hover {
        background-color: #f0f0f0; 
    }
    
    /* Expander Arrow Color (Make it match the green theme) */
    .st-emotion-cache-p2m900 .st-emotion-cache-1oyokq1 {
        color: #38761D !important;
    }
    
    </style>
    """, 
    unsafe_allow_html=True
)
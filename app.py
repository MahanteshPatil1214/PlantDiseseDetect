import streamlit as st
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import io

# --- 1. CONFIGURATION AND STYLING ---

st.set_page_config(
    page_title="Agri-Sense: Plant Disease Diagnostic",
    page_icon="🌱",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

def apply_custom_css():
    st.markdown("""
        <style>
        /* Modern UI Overrides */
        .stApp { background-color: #FAFAF5; }
        
        /* Sticky Header */
        .header-container {
            background-color: #38761D;
            color: white;
            padding: 1rem 2rem;
            border-radius: 0 0 15px 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }

        /* Result Cards */
        .status-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 8px solid #38761D;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        .confidence-text {
            font-size: 2rem;
            font-weight: bold;
            color: #28a745;
        }

        /* Responsive Buttons */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #38761D;
            color: white;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #2d5e17;
            border: none;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 2. RESOURCE LOADING ---

@st.cache_resource
def load_model():
    try:
        # Load model and handle the custom oneDNN messages internally
        model_path = "models/plant_disease_recog_model_pwp.keras" 
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

@st.cache_data
def load_disease_info():
    try:
        with open("plant_disease.json", 'r') as f:
            return json.load(f)
    except:
        # Fallback dataset for demonstration
        return {
            "0": {"name": "Tomato - Bacterial Spot", "cause": "Bacteria", "cure": "Copper spray"},
            "1": {"name": "Corn - Rust", "cause": "Fungus", "cure": "Fungicide"}
        }

model = load_model()
disease_data = load_disease_info()

# --- 3. LOGIC & PREPROCESSING ---

def preprocess_image(image_input):
    """Refined preprocessing for TensorFlow models."""
    img = Image.open(image_input).convert("RGB")
    img = img.resize((160, 160)) # Match your model's expected input
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Important: If your model was trained on [0,1], rescale here:
    # img_array = img_array / 255.0 
    return img_array

# --- 4. UI LAYOUT ---

st.markdown('<div class="header-container"><h1>🌱 Agri-Sense AI Diagnostic</h1></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Leaf Analysis")
    uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption="Target Image")
        # Store in session state for persistency across re-runs
        st.session_state['active_image'] = uploaded_file

with col2:
    st.subheader("🏥 Diagnostic Results")
    
    if uploaded_file:
        if st.button("🚀 Run Instant Diagnosis"):
            if model is None:
                st.error("Model not available.")
            else:
                with st.spinner("Our AI is scanning for pathogens..."):
                    # Process and Predict
                    processed_img = preprocess_image(uploaded_file)
                    prediction = model.predict(processed_img)
                    
                    # Logic to handle both JSON lists and JSON dictionaries
                    idx = np.argmax(prediction)
                    conf = np.max(prediction) * 100
                    
                    # Result extraction
                    if isinstance(disease_data, list):
                        res = disease_data[idx] if idx < len(disease_data) else None
                    else:
                        res = disease_data.get(str(idx))

                    if res:
                        st.markdown(f"""
                            <div class="status-card">
                                <p style="margin:0; color:#666;">DIAGNOSIS</p>
                                <h2 style="color:#C55A11; margin-top:0;">{res['name']}</h2>
                                <p style="margin:0; color:#666;">CONFIDENCE</p>
                                <span class="confidence-text">{conf:.1f}%</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        tab1, tab2 = st.tabs(["🦠 Pathology", "💊 Treatment Plan"])
                        with tab1:
                            st.write(res.get('cause', 'N/A'))
                        with tab2:
                            st.success(res.get('cure', 'Consult an expert.'))
                    else:
                        st.error("Diagnosis index not found in metadata.")
    else:
        st.info("Waiting for image upload to begin diagnosis.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 Agri-Sense AI. For educational use only. Always verify with local agricultural extension services.")
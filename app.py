import streamlit as st
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import io
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Agri-Sense AI",
    page_icon="🌱",
    layout="wide"
)

# --- 2. THEME & STYLING ---
def apply_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #FAFAF5; }
        .header-container {
            background-color: #38761D;
            color: white;
            padding: 1.5rem;
            border-radius: 0 0 15px 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            border-left: 8px solid #38761D;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .metric-label { color: #666; font-size: 14px; margin-bottom: 0; }
        .metric-val { font-size: 26px; font-weight: bold; color: #C55A11; margin-top: 0; }
        div.stButton > button {
            width: 100%;
            background-color: #38761D;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 3.5em;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 3. HELPER FUNCTIONS ---

def create_pdf(name, confidence, cause, cure):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(56, 118, 29)
    pdf.cell(0, 20, "Agri-Sense Diagnostic Report", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Diagnosis: {name}", ln=True)
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, f"Confidence Score: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Pathology & Cause:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, cause)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recommended Treatment:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, cure)
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_disease_info():
    try:
        with open("plant_disease.json", 'r') as f:
            return json.load(f)
    except:
        return [{"name": "Healthy Plant", "cause": "Normal growth.", "cure": "None needed."}]

model = load_model()
disease_data = load_disease_info()

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Agri-Sense AI")
    st.markdown("---")
    
    st.subheader("🧪 Example Gallery")
    st.info("No photo? Select a sample below:")
    
    example_options = {
        "Custom Upload": None,
        "Healthy Potato": "examples/plant.jpg",   # Matches your folder
        "Tomato Rust": "examples/plant1.jpeg",
    }
    selection = st.selectbox("Select Sample", list(example_options.keys()))
    
    if st.button("♻️ Reset All"):
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.markdown('<div class="header-container"><h1>🌱 Agri-Sense Plant Diagnostic</h1></div>', unsafe_allow_html=True)

col_img, col_res = st.columns([1.2, 1], gap="large")

# IMAGE PROCESSING LOGIC
final_image = None

with col_img:
    st.subheader("1. Leaf Input")
    
    # If "Custom Upload" is selected or no selection is made
    if selection == "Custom Upload":
        uploaded_file = st.file_uploader("Upload leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            final_image = Image.open(uploaded_file)
    else:
        # Load from Example Path
        try:
            final_image = Image.open(example_options[selection])
        except Exception as e:
            st.warning(f"Could not load example: {selection}. Ensure file exists in /examples/ folder.")

    if final_image:
        st.image(final_image, caption="Selected Image for Analysis", use_container_width=True)

with col_res:
    st.subheader("2. AI Diagnosis")
    
    if final_image:
        if st.button("🔎 Start Diagnosis"):
            if model:
                with st.spinner("Analyzing cell structures..."):
                    # Preprocessing (Ensuring RGB and resizing)
                    img_resized = final_image.convert("RGB").resize((160, 160))
                    img_array = tf.keras.utils.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Optional: Scaler (uncomment if model requires 0-1 range)
                    # img_array = img_array / 255.0
                    
                    # Prediction
                    predictions = model.predict(img_array)
                    idx = np.argmax(predictions)
                    conf = np.max(predictions) * 100
                    
                    # Extract info from JSON list
                    if idx < len(disease_data):
                        res = disease_data[idx]
                    else:
                        res = {"name": "Unknown", "cause": "Out of range", "cure": "N/A"}
                    
                    # DISPLAY CARDS
                    st.markdown(f"""
                        <div class="status-card">
                            <p class="metric-label">DIAGNOSIS</p>
                            <p class="metric-val">{res['name']}</p>
                            <hr style="margin: 10px 0;">
                            <p class="metric-label">AI CONFIDENCE</p>
                            <p class="metric-val" style="color:#28a745;">{conf:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📝 View Details", expanded=True):
                        st.markdown(f"**Cause:** {res.get('cause')}")
                        st.markdown(f"**Treatment:** {res.get('cure')}")
                    
                    # PDF EXPORT
                    report_data = create_pdf(res['name'], conf, res['cause'], res['cure'])
                    st.download_button(
                        label="📄 Download Treatment Report",
                        data=report_data,
                        file_name=f"Diagnosis_{res['name'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("AI Model not loaded. Check /models/ folder.")
    else:
        st.info("Waiting for image input. Use the sidebar for examples or upload your own.")

st.markdown("---")
st.caption("© 2025 Agri-Sense Farm Intelligence | AI for Sustainable Agriculture")
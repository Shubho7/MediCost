import pickle
import streamlit as st
import numpy as np
from xgboost import XGBRegressor

# Page configuration
st.set_page_config(
    page_title = "MediCost",
    layout = "centered",
    initial_sidebar_state = "collapsed"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 900px;
    }
    .stButton>button {
        width: 100%;
        background-color: #0303fc;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00047d;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #bb1900, #fd6f01, #ffb000);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
        width: 100%;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model/insurancemodel.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model not found.")
        return None

model = load_model()

# Header section
st.title("🏥 MEDICAL INSURANCE PREMIUM PRICE PREDICTOR")
st.markdown("### Get an instant estimate of your medical insurance premiums")
st.markdown("---")

# Input section
if model is not None:

    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "👤 Age",
            min_value = 0,
            max_value = 100,
            step = 1,
            format = "%d",
            help = "Enter your age"
        )
        
        bmi = st.number_input(
            "⚖️ BMI (Body Mass Index)",
            min_value = 0.0,
            max_value = 100.0,
            step = 0.1,
            format = "%.1f",
            help = "Enter your BMI"
        )
    
    with col2:
        smoker = st.selectbox(
            "🚬 Smoker",
            options = ["No", "Yes"],
            help = "Are you a smoker?"
        )
        
        children = st.number_input(
            "👶 Number of Children",
            min_value = 0,
            max_value = 10,
            step = 1,
            format = "%d",
            help = "Number of children covered under the insurance"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert smoker to binary
    smoker_binary = 1 if smoker == "Yes" else 0
    
    # Predict button
    if st.button("Predict Premium ➡️", use_container_width = True):
        # Prepare input data
        input_data = np.array([[int(age), float(bmi), smoker_binary, int(children)]], dtype=np.float64)
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            predicted_premium = float(prediction[0])
            
            # Display result
            st.markdown(f"""
                <div class = "prediction-box">
                    💰 Predicted Medical Insurance Premium Price: ${predicted_premium:,.2f}
                </div>
            """, unsafe_allow_html = True)
            
            st.balloons()
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'><i>© 2025 - Made By Group D Boys</i></p>",
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Unable to load the model.")


import streamlit as st
import pickle
import numpy as np
import random
import requests
from streamlit_lottie import st_lottie

# Function to load Lottie animation from URL
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie URL for AI Assistant (Modern AI Checkup animation)
LOTTIE_URL = "https://lottie.host/57a7d432-8418-4e89-a797-28e469e5d22f/oX61v6n268.json"
# Image URL for the header
HEADER_IMAGE_URL = "https://www.chatinbox.io/wp-content/uploads/chatbot-Healthcare-01.jpg"

# --- 1. Dummy Model and Encoder ---
# NOTE: You must replace these dummy classes with your actual 'pickle.load' code
# within the 'try...except' block below.
# This dummy code is only to allow the interface to run and display correctly.

# We are creating 132 mock symptoms as required by the model structure.
MOCK_SYMPTOMS = [
    'chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'joint_pain',
    'muscle_pain', 'cough', 'sore_throat', 'rash', 'abdominal_pain', 'chest_pain',
    'dizziness', 'loss_of_appetite', 'stomach_pain', 'diarrhoea', 'runny_nose',
    'redness_of_eyes', 'swollen_glands', 'back_pain', 'weight_loss', 'anxiety',
    'sunken_eyes', 'sweating', 'dehydration', 'constipation', 'skin_rash',
    'increased_appetite', 'puffy_face_and_eyes', 'blurred_and_distorted_vision',
    'throat_irritation', 'restlessness', 'lethargy', 'dark_urine', 'yellowish_skin',
    'foul_smell_of_urine', 'patches_in_throat', 'pain_during_bowel_movements',
    'spotting_urination', 'burning_micturition', 'loss_of_smell', 'shivering',
    'fast_heart_rate', 'irritability', 'mood_swings', 'weakness_in_limbs',
    'mucoid_sputum', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'cramps', 'bruising', 'acidity', 'ulcers_on_tongue', 'indigestion',
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_head',
    'lack_of_concentration', 'cold_hands_and_feets', 'palpitations', 'breathlessness',
    'excessive_hunger', 'dry_coughs', 'slurred_speech', 'small_dents_in_nails',
    'enlarged_thyroid', 'brittle_nails', 'prominent_veins_on_calf', 'swollen_legs',
    'extra_marital_contacts', 'receiving_blood_transfusion', 'history_of_alcohol_consumption',
    'bell_s_palsy', 'altered_sensorium', 'unsteadiness', 'passage_of_gases',
    'internal_itching', 'toxic_look_(typhos)', 'depression', 'change_in_voice',
    'continuous_sneezing', 'silver_like_dusting', 'blister_on_tongue', 'dischromic _patches',
    'scurring', 'polyuria', 'hip_joint_pain', 'knee_pain', 'phlegm', 'toxic_look_typhos',
    'family_history', 'red_spots_over_body', 'bleeding_gums', 'pus_filled_pimples',
    'blackheads', 'skin_peeling', 'nodal_skin_eruptions', 'yellow_crust_over_eyes',
    'watery_eyes', 'malaise', 'mild_fever', 'hip_joint_pain', 'swelling_of_stomach',
    'distention_of_abdomen', 'loss_of_balance', 'uncontrolled_bladder',
    'continuous_feel_of_urine', 'itching', 'internal_itching', 'burning_in_throat',
    'passage_of_gases', 'foul_smell_of_stool', 'excess_sweating', 'yellow_urine',
    'acute_liver_failure', 'coma', 'stomach_bleeding', 'internal_bleeding',
    'blood_in_sputum', 'irregular_heart_beat', 'chest_tightness', 'paralysis_of_limbs',
    'fluid_overload', 'swelling_in_calves', 'painful_walking', 'small_rash',
    'excessive_thirst', 'visual_disturbances', 'unusual_smell', 'difficulty_in_swallowing',
    'prognosis' # Assuming 'prognosis' is often the last feature or just adding to 132
]
# Ensure we have exactly 132 for model compatibility simulation
if len(MOCK_SYMPTOMS) < 132:
    MOCK_SYMPTOMS.extend([f'symptom_{i+100}' for i in range(132 - len(MOCK_SYMPTOMS))])
elif len(MOCK_SYMPTOMS) > 132:
    MOCK_SYMPTOMS = MOCK_SYMPTOMS[:132]

# Dummy Model Class
class MockModel:
    def __init__(self, n_features=132):
        self.n_features_in_ = n_features
    def predict(self, X):
        # Dummy logic: returns a random index based on the number of symptoms
        if np.sum(X[0]) == 0:
            return [0] # No symptoms = Healthy
        if np.sum(X[0]) < 3:
            return [1] # Mild symptoms
        if np.sum(X[0]) < 6:
            return [2] # Moderate symptoms
        return [3] # Severe symptoms

# Dummy Label Encoder Class
class MockLabelEncoder:
    def inverse_transform(self, labels):
        # Dummy disease names (kept mixed language for assistant feel)
        disease_map = {
            0: 'Aap Bilkul Swasth Hain (You are perfectly healthy)',
            1: 'Aam Zukaam / Maamooli Bukhaar (Common Cold / Mild Fever)',
            2: 'Typhoid ya Dengue ki Shanka (Suspicion of Typhoid or Dengue)',
            3: 'Kripya Turant Doctor se Milen (Please consult a doctor immediately)'
        }
        return [disease_map.get(labels[0], 'Asamyakt Rog (Unclassified Disease)')]

# --- 2. Model Loading ---
@st.cache_resource
def load_models():
    """Loads the model and encoder."""
    try:
        # NOTE: Use your actual file paths here!
        # rf = pickle.load(open('minor project/best_rf_model.pkl', 'rb'))
        # le = pickle.load(open('minor project/rf_label_encoder.pkl', 'rb'))

        # Loading Dummy Model for execution
        rf = MockModel(n_features=132)
        le = MockLabelEncoder()

        return rf, le

    except Exception as e:
        # Error handling if loading fails (e.g., file not found)
        st.error(f"Error loading models. Ensure your .pkl files are correct in the 'minor project/' folder. \nError: {e}")
        # Return dummy objects if model loading failed
        return MockModel(n_features=132), MockLabelEncoder()

# --- 3. Prediction Function ---
def predict_disease(rf_model, le_encoder, selected_symptoms, all_symptoms):
    """Predicts the disease based on selected symptoms."""
    if not selected_symptoms:
        # If no symptoms are selected
        return le_encoder.inverse_transform([0])[0], 0

    # Create the input vector with 132 features
    input_vector = np.zeros(len(all_symptoms), dtype=int)
    symptom_to_index = {symptom: i for i, symptom in enumerate(all_symptoms)}

    # Set '1' for selected symptoms
    for symptom in selected_symptoms:
        if symptom in symptom_to_index:
            index = symptom_to_index[symptom]
            input_vector[index] = 1

    # Prediction
    try:
        pred_label = rf_model.predict([input_vector])[0]
        pred_disease = le_encoder.inverse_transform([pred_label])[0]
        return pred_disease, len(selected_symptoms)
    except Exception as e:
        # Handle model prediction errors
        st.error(f"Prediction Error: {e}")
        return "Model Error", len(selected_symptoms)

# --- 4. Streamlit UI (Web Interface) ---

# Page Configuration (for a classic and wide layout)
st.set_page_config(
    page_title="AI Health Assistant | AI Health Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Call function to load models
rf, le = load_models()
ALL_SYMPTOMS = MOCK_SYMPTOMS # In actual code, you should load this list from your dataset

# Classic CSS Styling
st.markdown("""
<style>
    /* For a classic and appealing interface */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #1e1e1e;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #2c3e50; /* Dark blue/gray color */
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: 700;
    }
    .stSelectbox label, .stMultiSelect label {
        font-size: 1.1em;
        font-weight: 600;
        color: #34495e; /* Medium dark color */
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #3498db; /* Blue color */
        color: white;
        border-radius: 12px;
        padding: 0.6em 2em;
        font-size: 1.2em;
        font-weight: 600;
        border: 2px solid #2980b9;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:hover {
        background-color: #2980b9;
        border-color: #3498db;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    /* Chat/Assistant Container Styling */
    .assistant-message {
        background-color: #ecf0f1; /* Light blue/gray */
        border-radius: 15px;
        padding: 15px 20px;
        margin-top: 20px;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .assistant-title {
        color: #3498db;
        font-weight: 700;
        font-size: 1.3em;
        margin-bottom: 5px;
    }
    /* Note styling (black background, white text) */
    .note-style {
        background-color: #1e1e1e !important; /* Black background */
        color: white !important; /* White text */
        border-left: 5px solid #3498db !important; /* Keep blue border for emphasis */
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
    }
    /* Custom Footer Styling */
    .footer-credit {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f0f2f6;
        color: #7f8c8d;
        font-size: 0.9em;
        border-top: 1px solid #bdc3c7;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTENT (Updated) ---
lottie_ai = load_lottieurl(LOTTIE_URL)

with st.sidebar:
    # FIX: Only display Lottie animation if the data was loaded successfully (not None)
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_animation")
    else:
        # Fallback if animation fails to load
        st.markdown("<p style='text-align:center; font-size: 50px;'>🤖</p>", unsafe_allow_html=True)
        
    st.markdown("""
    ### 🤝 Your Personal Health Assistant
    
    Hello! I'm your AI Checkup Assistant. My goal is to help you identify potential health issues based on the symptoms you provide using a Machine Learning model.
    
    **Remember:** I am an assistive tool, not a doctor. Always consult a medical professional for accurate diagnosis and treatment.
    """)
    st.markdown("---")
    st.markdown("""
    ### Disclaimer
    This application uses a machine learning model for prediction and is **not a substitute for professional medical advice**. Always consult a doctor for diagnosis and treatment.
    """)

# Main Title
st.title("🩺 AI Health Assistant 🤖")

# --- HEADER IMAGE INSERTION ---
# Fix: Replaced deprecated 'use_column_width="auto"' with 'use_container_width=True'
st.image(HEADER_IMAGE_URL, use_container_width=True, caption="AI-Powered Medical Chatbot Assistant")

st.markdown("""
<p style="text-align: center; font-size: 1.2em; color: #7f8c8d;">
Hello! Please select your symptoms from the options below. I will use your symptoms to predict the potential disease.
</p>
""", unsafe_allow_html=True)

# --- SIMPLE CHATBOT FEATURE (FIXED) ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        # Using a simple container for chat visualization
        st.markdown(f'<div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: right;">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #f0fff4; padding: 10px; border-radius: 10px; margin-bottom: 5px;">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)

# FIX: Wrap chat input in a form to prevent continuous rerunning after submission.
# `clear_on_submit=True` automatically resets the text input after submission.
with st.form("chat_form", clear_on_submit=True):
    # Use a generic key for the input field inside the form
    chat_input = st.text_input("Send a message to the Assistant (Try typing 'hi'):", key="chat_input_text", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send ⬆️")

    if submitted and chat_input:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": chat_input})
        
        # Process assistant response
        response = ""
        if chat_input.lower().strip() == "hi":
            response = "Hello! I am your AI Health Assistant. You can use the form below to select your symptoms for a preliminary checkup. How can I help you today?"
        else:
            response = "I am currently designed to help you with the symptom prediction form below. Please use the selection boxes to proceed with your checkup."
        
        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # NOTE: Form submission triggers a single rerun automatically. No st.rerun() needed here.

# --- END OF SIMPLE CHATBOT FEATURE ---

# Main input form in a container
with st.container():
    st.markdown("---")
    
    # Column layout (to center the input)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Multi-select box for choosing symptoms (searchable)
        selected_symptoms = st.multiselect(
            "Select Your Symptoms:",
            options=ALL_SYMPTOMS,
            default=[],
            help="You can select multiple symptoms."
        )

        # Prediction Button
        if st.button("🔍 Predict Disease", use_container_width=True):
            # Show results only if symptoms are selected
            if selected_symptoms:
                # Call prediction function
                with st.spinner('Analyzing your symptoms...'):
                    predicted_disease, symptom_count = predict_disease(rf, le, selected_symptoms, ALL_SYMPTOMS)
                
                # Use custom black/white styling for the prediction output
                st.markdown('<div class="assistant-message note-style">', unsafe_allow_html=True)
                st.markdown('<div class="assistant-title" style="color: white !important;">🤖 AI Assistant\'s Prediction:</div>', unsafe_allow_html=True)
                
                # Display result
                st.write(f"""
                    Based on the **{symptom_count}** symptoms you selected, 
                    our prediction is that the potential disease could be **{predicted_disease}**.
                """)
                
                # Important advice (Adjusting these to use text color styling within the note-style div)
                if "Doctor" in predicted_disease or "Shanka" in predicted_disease:
                    # CHANGED color from 'yellow' to a brighter, high-contrast orange-red
                    st.markdown("<p style='color: #FF5733;'>⚠️ This might be a serious sign. You should **consult a qualified doctor** without any delay.</p>", unsafe_allow_html=True)
                elif "Swasth" in predicted_disease:
                     st.balloons()
                     st.markdown("<p style='color: #4CAF50;'>🎉 Good news! Your symptoms do not indicate any serious illness. Rest if you feel unwell.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #B3E5FC;'>💡 **Note:** This is only an estimation from a machine learning model. Always consult a **Medical Doctor** for an accurate diagnosis.</p>", unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning("Please select at least one symptom to make a prediction.")
    
    st.markdown("---")

# --- FOOTER CREDIT (New) ---
st.markdown("""
<div class="footer-credit">
    <p>Designed by JIMMY THOMAS and MOHD AYAN</p>
</div>
""", unsafe_allow_html=True)




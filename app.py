import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from datetime import time
import matplotlib.pyplot as plt

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./my_saved_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./my_saved_model')

# Label encoder to map class IDs back to class names (crime descriptions)
crime_classes = ['BATTERY - SIMPLE ASSAULT', 'BURGLARY', 'BURGLARY FROM VEHICLE', 'THEFT PLAIN - PETTY ($950 & UNDER)', 'VEHICLE - STOLEN']

# Streamlit page configuration with theme changes
st.set_page_config(page_title="Crime Class Predictor", page_icon="🚨", layout="centered")

# Custom CSS for a red theme with better visibility for headers
st.markdown("""
    <style>
    body {
        background-color: #1b1b1b;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #b33030;
        color: white;
    }
    .stNumberInput, .stSelectbox, .stTimeInput {
        background-color: #2c2c2c;
        border: 1px solid #ff4b4b;
        border-radius: 8px;
        padding: 8px;
        color: white;
    }
    .stNumberInput input, .stSelectbox select, .stTimeInput input {
        color: white;
    }
    .stNumberInput input, .stSelectbox select {
        font-size: 16px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for branding and about information
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=efW10Mdj2uRb&format=png&color=000000", width=150)  # Replace with your logo URL
    st.title("Crime Class Predictor")
    st.markdown("This app predicts the probability of different crime classes based on the provided area code, time, and day of the week.")
    st.markdown("### About")
    st.write("The model is trained on historical crime data and uses advanced NLP techniques to predict crime probabilities.")

# Main application content
st.header("Predict the Crime Class")
st.write("Enter the required details below to get the crime class probabilities:")

# Input layout in columns
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Enter Area Code", min_value=1, max_value=22, step=1)
    
with col2:
    selected_time = st.time_input("Select the Time", value=time(12, 0))
    
with col3:
    day_of_week = st.selectbox("Select the Day", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Display separator for better visuals
st.markdown("---")

# Combine input features into a single string as required by the model
combined_features = f"Area: {area} Hour: {selected_time} DayOfWeek: {day_of_week}"

# Button for predictions
if st.button('Predict Crime Class'):
    with st.spinner("Analyzing data and predicting..."):
        # Tokenize the input
        inputs = tokenizer(combined_features, return_tensors="pt", truncation=True, padding=True)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

    # Display the probabilities for each class
    st.success("Prediction Complete!")
    st.write("### Crime Class Probabilities:")

    # Create a bar chart using Matplotlib with dark background and visible labels
    fig, ax = plt.subplots(figsize=(8, 5))
    crime_names = crime_classes
    probs = probabilities * 100  # Convert to percentages
    bars = ax.barh(crime_names, probs, color='#ff4b4b')

    # Add percentage labels on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center', fontsize=12, color='#ffffff')

    # Set chart background and text colors
    fig.patch.set_facecolor('#1b1b1b')  # Dark background for the figure
    ax.set_facecolor('#1b1b1b')         # Dark background for the chart area
    ax.set_xlabel('Probability (%)', fontsize=14, color='#ffffff')
    ax.set_title('Crime Class Prediction Probabilities', fontsize=16, color='#ff4b4b')
    
    # Ensure category labels are also white and easy to read
    ax.tick_params(axis='y', colors='#ffffff', labelsize=12)
    ax.tick_params(axis='x', colors='#ffffff')  # Set x-tick colors to white for better contrast

    plt.xlim([0, 100])

    # Display the chart in Streamlit
    st.pyplot(fig)

    # Safety Tips based on the crime class with the highest probability
    st.subheader("Safety Tips")

    max_crime_class = crime_classes[probabilities.argmax()]

    if max_crime_class == 'VEHICLE - STOLEN':
        st.write("🚗 **Tip:** Always park in well-lit areas, lock your car, and never leave valuables visible inside.")
    elif max_crime_class == 'BURGLARY':
        st.write("🏠 **Tip:** Ensure all doors and windows are locked. Consider installing a security system and motion-sensor lights.")
    elif max_crime_class == 'BURGLARY FROM VEHICLE':
        st.write("🚘 **Tip:** Never leave your car unlocked, and remove all valuable items when parking.")
    elif max_crime_class == 'BATTERY - SIMPLE ASSAULT':
        st.write("👮 **Tip:** Avoid isolated areas late at night. Always stay alert and keep your phone close.")
    elif max_crime_class == 'THEFT PLAIN - PETTY ($950 & UNDER)':
        st.write("💼 **Tip:** Keep an eye on your personal belongings in crowded areas. Secure your bags and wallets in public.")

# Footer
st.markdown("""
    <div class="footer">
        Note: This prediction is based on historical data and machine learning models. It is not a definitive outcome but a probability-based prediction.
    </div>
    """, unsafe_allow_html=True)

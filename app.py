import streamlit as st
import joblib
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="🎭",
    layout="centered"
)

# Function to load the model and vectorizer
# Using st.cache_resource to load them only once and share across sessions for performance
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        return None, None

def main():
    st.title("🎭 Emotion Analyzer")
    st.markdown("Check whether a given sentence or review expresses **Fear**, **Anger**, or **Joy** using a trained machine learning model.")
    
    st.markdown("---")

    # Load machine learning components
    model, vectorizer = load_model_and_vectorizer()

    if model is None or vectorizer is None:
        st.warning("⚠️ **Model files not found!** Please ensure `model.pkl` and `vectorizer.pkl` are located in the same directory as this app.")
        st.stop()  # Stop execution until files are provided

    # Text input from the user
    user_input = st.text_area(
        label="Enter your text here:", 
        placeholder="e.g., I absolutely loved this product! Highly recommended.",
        height=150
    )

    # Prediction button
    if st.button("Predict Emotion", type="primary", use_container_width=True):
        if not user_input.strip():
            # Handle empty input
            st.error("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # 1. Transform the text using the loaded CountVectorizer
                    text_vector = vectorizer.transform([user_input])
                    
                    # 2. Predict sentiment using the loaded SVM model
                    prediction = model.predict(text_vector)[0]
                    
                    # 3. Determine and display result
                    # Model outputs numeric labels: 0=Fear, 1=Anger, 2=Joy
                    pred_val = int(prediction)
                    
                    if pred_val == 0:
                        st.warning("### Emotion: Fear 😨")
                    elif pred_val == 1:
                        st.error("### Emotion: Anger 😡")
                    elif pred_val == 2:
                        st.success("### Emotion: Joy 😊")
                    else:
                        st.info(f"### Predicted Emotion (Unknown Label): {prediction}")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()

"""This Module is the main entry point for the Streamlit application."""

# main.py
import pandas as pd
import torch
import streamlit as st
from scripts import (
    convert_inputs,
    FEATURE_NAMES,
    load_model,
    load_feature_scaler,
    load_label_scaler,
    CATEGORY_MAPPING,
    GENDER_MAPPING,
    STATE_MAPPING,
)

# Main Loop
# Call this function, during script execution; Main script entry point
if __name__ == "__main__":
    st.title("Agent")
    st.subheader("Check For Credit Card Fraud", divider=True)

    feature_scaler = load_feature_scaler()
    label_scaler = load_label_scaler()

    agent = load_model()

    with st.form("my_form"):

        categories = CATEGORY_MAPPING.keys()
        states = STATE_MAPPING.keys()
        genders = GENDER_MAPPING.keys()

        st.write("Please provide the following information:")

        # Some links for data validation: https://www.baeldung.com/java-geo-coordinates-validation
        user_inputs = {
            "category": st.selectbox("Category", categories),
            "amt": st.number_input(
                "Amount", min_value=1.0, max_value=30000.00, value=25.0, key="amt"
            ),
            "gender": st.radio("Gender", genders),
            "state": st.selectbox("From what State is the Card Owner from?", states),
            "lat": st.slider("Enter the lattitude of the card owner", -90.0, 90.0, 20.0, step=0.01),
            "long": st.slider(
                "Enter the longitude of the card owner", -180.0, 180.0, -165.0, step=0.01
            ),
            "city_pop": st.slider(
                "Enter the city population of the card owner", 23.0, 2906700.0, 40000.0, step=1.0
            ),
            "merch_lat": st.slider("Merchant Latitude", -90.0, 90.0, 20.0, step=0.01),
            "merch_long": st.slider("Merchant Longitude", -180.0, 180.0, -165.0, step=0.01),
        }

        # Process the inputs and sample from the model
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            # Convert inputs to the correct format using the expansion operator
            converted_inputs = convert_inputs(*user_inputs.values())

            # Create a DataFrame for the scaler using the feature names to prevent warnings
            input_df = pd.DataFrame([converted_inputs], columns=FEATURE_NAMES)

            # Transform input tensor by scaling the inputs using the pre-fitted scaler
            inputs = feature_scaler.transform(input_df)

            inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert to tensor

            unnormalized_pred = agent.get_prediction(inputs)

            prediction_index = torch.argmax(unnormalized_pred)
            prediction_label = (
                "FRAUD" if prediction_index == 1 else "NOT FRAUD"
            )  # Map the prediction index to a label

            st.write("---")  # Separator

            # --- Streamlit Output ---
            st.subheader("Classification Result:")

            if prediction_label == "FRAUD":
                st.error(f"Prediction: {prediction_label} 🚨")  # Red for alert
            else:
                st.success(f"Prediction: {prediction_label} ✅")

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
        user_inputs = []

        st.write("Please provide the following information:")
        # User inputs
        category = st.selectbox(
            "What is the category of the transaction?",
            (
                "entertainment",
                "food_dining",
                "gas_transport",
                "grocery_net",
                "grocery_pos",
                "health_fitness",
                "home",
                "kids_pets",
                "misc_net",
                "misc_pos",
                "personal_care",
                "shopping_net",
                "shopping_pos",
                "travel",
            ),
        )
        user_inputs.append(category)

        amt = st.number_input(
            "What is the amount of the transaction?",
            min_value=1.0,
            max_value=30_000.00,
            value=25.0,
            key="amt",
        )
        user_inputs.append(amt)

        gender = st.radio("Choose an option", ["M", "F"])
        user_inputs.append(gender)

        state = st.selectbox(
            "From what State is the Card Owner from?",
            (
                "AK",
                "AL",
                "AR",
                "AZ",
                "CA",
                "CO",
                "CT",
                "DC",
                "DE",
                "FL",
                "GA",
                "HI",
                "IA",
                "ID",
                "IL",
                "IN",
                "KS",
                "KY",
                "LA",
                "MA",
                "MD",
                "ME",
                "MI",
                "MN",
                "MO",
                "MS",
                "MT",
                "NC",
                "ND",
                "NE",
                "NH",
                "NJ",
                "NM",
                "NV",
                "NY",
                "OH",
                "OK",
                "OR",
                "PA",
                "RI",
                "SC",
                "SD",
                "TN",
                "TX",
                "UT",
                "VA",
                "VT",
                "WA",
                "WI",
                "WV",
                "WY",
            ),
        )
        user_inputs.append(state)

        # Some links for data validation: https://www.baeldung.com/java-geo-coordinates-validation
        lat = st.slider(
            label="Enter the lattitude of the card owner",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            step=0.01,
        )
        user_inputs.append(lat)

        long = st.slider(
            label="Enter the longitude of the card owner",
            min_value=-180.0,
            max_value=180.0,
            value=-165.0,
            step=0.01,
        )
        user_inputs.append(long)

        city_pop = st.slider(
            label="Enter the city population of the card owner",
            min_value=23.0,
            max_value=2_906_700.0,
            value=40_000.0,
            step=1.0,
        )
        user_inputs.append(city_pop)

        merch_lat = st.slider(
            label="Enter the lattitude of the store/vendor/purchase",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            step=1.0,
        )
        user_inputs.append(merch_lat)

        merch_long = st.slider(
            label="Enter the longitude of the store/vendor/purchase",
            min_value=-180.0,
            max_value=180.0,
            value=-165.0,
            step=0.01,
        )
        user_inputs.append(merch_long)

        # Process the inputs and sample from the model
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            # Convert inputs to the correct format using the expansion operator
            converted_inputs = convert_inputs(*user_inputs)

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

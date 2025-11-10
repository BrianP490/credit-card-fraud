"""This Module is the main entry point for the Streamlit application."""

# main.py
import sys
import json
import pandas as pd
import torch
import joblib
import streamlit as st
from scripts import (
    Agent,
    convert_inputs,
    MODEL_WEIGHTS_FULL_PATH,
    CONFIG_PATH,
    FEATURE_SCALER_PATH,
    FEATURE_NAMES,
)

# Main Loop
# Call this function, during script execution; Main script entry point
if __name__ == "__main__":
    st.title("Agent")
    st.subheader("Check For Credit Card Fraud", divider=True)

    # Load model weights
    try:
        model_weights = torch.load(MODEL_WEIGHTS_FULL_PATH, weights_only=True)
        print(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")
    except FileNotFoundError:
        print(
            f"❌ Model Weights file not found at '{MODEL_WEIGHTS_FULL_PATH}'. "
            ""
            "Please ensure the file exists or fix path to file."
        )
        sys.exit(1)

    # Load configuration file
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(
            f"❌ Configuration file not found at '{CONFIG_PATH}'. "
            "Please ensure the file exists or fix path to file."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        sys.exit(1)

    # Load feature scaler
    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        print(f"✅ Feature Scaler loaded successfully from {FEATURE_SCALER_PATH}")
    except FileNotFoundError:
        print(
            f"❌ Configuration file not found at '{FEATURE_SCALER_PATH}'. "
            "Please ensure the file exists or fix path to file."
        )
        sys.exit(1)

    label_scaler = None
    MODEL_CONFIG = config.get("model", {})

    try:
        agent = Agent(cfg=MODEL_CONFIG)  # Create agent instance
        agent.load_state_dict(state_dict=model_weights)
    except RuntimeError as e:
        print(f"❌ A runtime error occurred while creating model or loading model weights: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Model weights file not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing key in model configuration: {e}")
        sys.exit(1)

    agent.eval().to("cpu")

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
        lat = st.slider(min_value=-90.0, max_value=90.0, value=20.0, step=0.01)
        user_inputs.append(lat)

        long = st.slider(min_value=-180.0, max_value=180.0, value=-165.0, step=0.01)
        user_inputs.append(long)

        city_pop = st.slider(min_value=23.0, max_value=2_906_700.0, value=40_000.0, step=0.01)
        user_inputs.append(city_pop)

        merch_lat = st.slider(min_value=-90.0, max_value=90.0, value=20.0, step=1.0)
        user_inputs.append(merch_lat)

        merch_long = st.slider(min_value=-180.0, max_value=180.0, value=-165.0, step=0.01)
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
            pred = label_scaler.inverse_transform([[unnormalized_pred]])[
                0, 0
            ]  # Un-normalize the prediction
            st.success(f"Agent Predicts: **{pred:.2f}**")

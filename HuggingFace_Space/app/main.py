"""This Module is the main entry point for the Streamlit application."""

import logging
import pandas as pd
import torch
import streamlit as st
from scripts import (
    load_config,
    setup_logger,
    convert_inputs,
    FEATURE_NAMES,
    load_model,
    load_feature_scaler,
    load_label_scaler,
    INPUT_METADATA,
)

# Configure Logger
logging_config = load_config()["logging"]

logger = setup_logger(logging_config)


# Main script entry point
if __name__ == "__main__":

    st.title("Agent")

    st.subheader("Check For Credit Card Fraud", divider=True)

    # Load Scalers
    feature_scaler = load_feature_scaler(logger)

    label_scaler = load_label_scaler(logger)

    # Build the model architecture and load in the model weights
    agent = load_model(logger)

    # Create a Streamlit form to take in user inputs
    with st.form("my_form"):

        st.write("Please provide the following information:")

        user_inputs = {}  # Variable to store the user's input

        # ----------------------------------------------------
        # Loop over metadata to create widgets
        # ----------------------------------------------------
        for key, meta in INPUT_METADATA.items():
            title = meta["title"]
            widget_type = meta["widget_type"]

            # Create widgets dynamically
            if widget_type == "selectbox":
                user_inputs[key] = st.selectbox(title, meta["options"])

            elif widget_type == "radio":
                user_inputs[key] = st.radio(
                    title,
                    options=meta["options"],
                    horizontal=meta.get("horizontal", None),
                    index=meta.get("preselected_index", 0),
                )

            elif widget_type == "number_input":
                user_inputs[key] = st.number_input(
                    title,
                    min_value=meta["min_value"],
                    max_value=meta["max_value"],
                    value=meta["value"],
                    key=key,  # Use the key from metadata
                    step=meta.get("step", None),
                )

            elif widget_type == "slider":
                user_inputs[key] = st.slider(
                    title,
                    meta["min_value"],
                    meta["max_value"],
                    meta["value"],
                    step=meta.get("step", None),
                )

            # ----------------------------------------------------
            # Add other widget types as needed (e.g., st.text_input)
            # ----------------------------------------------------

        # Process the inputs and sample from the model
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            # DEBUGGING: Log the raw inputs dictionary before processing
            # logger.info(f"User submitted prediction request with inputs: {user_inputs}")

            # Convert inputs to the correct format using the keyword expansion operator
            converted_inputs = convert_inputs(**user_inputs)

            # Create a DataFrame for the scaler using the feature names to prevent warnings
            input_df = pd.DataFrame([converted_inputs], columns=FEATURE_NAMES)

            # Transform input tensor by scaling the inputs using the pre-fitted scaler
            inputs = feature_scaler.transform(input_df)

            inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert to tensor

            # ----------------------------------------------------
            # Modify for Each Machine Learning Task
            # ----------------------------------------------------

            output = agent.get_prediction(inputs)

            # Unscale outputs with the label scaler if necessary

            prediction_index = torch.argmax(output)
            prediction_label = (
                "FRAUD" if prediction_index == 1 else "NOT FRAUD"
            )  # Map the prediction index to a label

            st.write("---")  # Separator for formatting

            # --- Streamlit Output ---
            st.subheader("Classification Result:")

            if prediction_label == "FRAUD":
                st.error(f"Prediction: {prediction_label} 🚨")  # Red for alert
            else:
                st.success(f"Prediction: {prediction_label} ✅")

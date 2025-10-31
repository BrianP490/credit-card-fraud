# main.py
import torch
import streamlit as st
from scripts import Agent, ModuleLayer
import os
import pandas as pd
import joblib

st.title("Agent")

st.subheader("Check For Credit Card Fraud", divider=True)

# Same feature order names and order as during the data pipeline during model training
feature_names = [
    "category",
    "amt",
    "gender",
    "state",
    "lat",
    "long",
    "city_pop",
    "merch_lat,merch_long",
]

# Configuration for the Agent model from original training ('/configs/config.json')
cfg = {
    "in_dim": len(feature_names),  # Number of Features as input
    "intermediate_dim": 128,
    "out_dim": 2,
    "num_blocks": 12,  # Number of reapeating Layer Blocks
    "dropout_rate": 0.1,  # Rate for dropout layer
}


def convert_inputs(*args) -> list:
    """Convert user inputs into a list of features for the model.
    Args:
        *args: Variable length argument list containing user inputs in the following order:
            category (tbd): Category of the transaction (acceptability tbd).
            amt (tbd): Amount of the transaction (tbd).
            gender (tbd): Gender of the card owner (tbd).
            state (tbd): State of the card owner (tbd).
            lat (tbd): Lattitude of the card owner's home (tbd).
            long (tbd): Longitude of the card owner's home (tbd).
            city_pop (tbd): City population of the card owner (tbd).
            merch_lat (tbd): Lattitude of the merchant (tbd).
            merch_long (tbd): Longitude of the merchant (tbd).
    Returns:
        features: A list of converted features ready for model input.
    """
    features = []

    try:

        # category
        category = args[0]
        if not (0 <= category <= 13):
            raise ValueError("category out of range.")
        features.append(float(category))

        # amt
        amt = args[1]
        if not (0 <= amt <= 30, 000):
            raise ValueError("amt out of range.")
        features.append(float(amt))

        # gender
        gender = args[2]
        if not isinstance(gender, str):
            raise ValueError("gender must be a string.")
        features.append(1.0 if gender.lower() == "male" else 0.0)

        # state
        state = args[0]
        if not isinstance(state, str):
            raise ValueError("state must be a string.")
        features.append(None)  # TBD; need to use a dictonary to convert to float

        # lat
        lat = args[1]
        if not isinstance(lat, float):
            raise ValueError("lat must be a float.")
        features.append(lat)

        # long
        long = args[2]
        if not isinstance(long, float):
            raise ValueError("long must be a float.")
        features.append(long)

        # city_pop
        city_pop = args[0]
        if not (1 <= city_pop <= 3_000_000):
            raise ValueError("city_pop out of range.")
        features.append(float(city_pop))

        # merch_lat
        merch_lat = args[1]
        if not isinstance(merch_lat, float):
            raise ValueError("merch_lat must be a float.")
        features.append(merch_lat)

        # merch_long
        merch_long = args[2]
        if not isinstance(merch_long, float):
            raise ValueError("merch_long must be a float.")
        features.append(merch_long)
    except Exception as e:
        st.error(f"Error in input conversion: {e}")

    return features


agent = Agent(cfg)  # Create agent instance

# Dynamically create the path to the model's weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory of current running file

weights_file = os.path.join(
    BASE_DIR, "model_weights", "my-model.pt"
)  # create the full path to the model weights
# Create the full path to the scalers
features_scaler_file = os.path.join(BASE_DIR, "scalers", "feature-scaler.joblib")
label_scaler_file = os.path.join(BASE_DIR, "scalers", "label-scaler.joblib")

features_scaler = joblib.load(features_scaler_file)  # Load feature scaler
# label_scaler = joblib.load(label_scaler_file)  # Load label scaler
label_scaler = None


try:
    agent.load_state_dict(
        torch.load(weights_file, weights_only=True)
    )  # Load the agent's model weights
except Exception as e:
    st.error(f"Error loading model weights: {e}")


with st.form("my_form"):
    st.write("Please provide the following information:")

    # User inputs
    age = st.slider("How old are you?", min_value=18, max_value=64, value=32, key="age")
    gender = st.radio("Choose an option", ["Male", "Female"])
    bmi = st.number_input(
        "What is your BMI?", min_value=15.96, max_value=53.13, value=25.0, key="bmi"
    )

    # Process the inputs and sample from the model
    submitted = st.form_submit_button("Get Prediction")
    if submitted:
        # Create a list of features from the user's inputs
        converted_inputs = convert_inputs(age, sex, bmi)

        # Create a DataFrame for the scaler using the feature names to prevent warnings
        input_df = pd.DataFrame([converted_inputs], columns=feature_names)

        # Transform input tensor
        inputs = features_scaler.transform(input_df)  # Scale the inputs using the pre-fitted scaler

        inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert to tensor

        unnormalized_pred = agent.get_prediction(inputs)
        pred = label_scaler.inverse_transform([[unnormalized_pred]])[
            0, 0
        ]  # Un-normalize the prediction
        st.success(f"Agent Predicts: **{pred:.2f}**")

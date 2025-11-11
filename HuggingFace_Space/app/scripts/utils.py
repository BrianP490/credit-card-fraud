"""This module contains utility functions for input conversion and validation."""

import json
import joblib
import streamlit as st
import torch

from .consts import (
    FEATURE_NAMES,
    CATEGORY_MAPPING,
    GENDER_MAPPING,
    STATE_MAPPING,
    MODEL_WEIGHTS_FULL_PATH,
    CONFIG_PATH,
    FEATURE_SCALER_PATH,
)
from .model import Agent


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
    features = []  # Create empty list to store all the features

    try:
        # category
        category = args[0]
        if not isinstance(category, str):
            raise ValueError("category must be a string.")

        category = CATEGORY_MAPPING.get(category, None)

        if category is not None:
            features.append(category)

        # amt
        amt = args[1]
        if not (1.0 <= amt <= 30_000.00):
            raise ValueError("amt out of range.")
        features.append(float(amt))

        # gender
        gender = args[2]
        if not isinstance(gender, str):
            raise ValueError("gender must be a string.")

        gender = GENDER_MAPPING.get(gender, None)

        if gender is not None:
            features.append(gender)

        # state
        state = args[3]
        if not isinstance(state, str):
            raise ValueError("state must be a string.")

        state = STATE_MAPPING.get(state, None)

        if state is not None:
            features.append(state)

        # lat
        lat = args[4]
        if not isinstance(lat, float):
            raise ValueError("lat must be a float.")
        features.append(lat)

        # long
        long = args[5]
        if not isinstance(long, float):
            raise ValueError("long must be a float.")
        features.append(long)

        # city_pop
        city_pop = args[6]
        if not (23.0 <= city_pop <= 2_906_700.0):
            raise ValueError("city_pop out of range.")
        features.append(float(city_pop))

        # merch_lat
        merch_lat = args[7]
        if not isinstance(merch_lat, float):
            raise ValueError("merch_lat must be a float.")
        features.append(merch_lat)

        # merch_long
        merch_long = args[8]
        if not isinstance(merch_long, float):
            raise ValueError("merch_long must be a float.")
        features.append(merch_long)

        if len(features) != len(FEATURE_NAMES):
            raise ValueError("Model Missing Input Feature(s). Please check.")

    except IndexError as e:
        st.error(f"Error in indexing inputs: {e}")
    except TypeError as e:
        st.error(f"Type error in input conversion: {e}")
    except ValueError as e:
        st.error(f"Value error in input conversion: {e}")

    return features


@st.cache_data
def load_config():
    """Loads configuration file using global variable. Optimized using streamlit caching.
    Args:
        N/A
    Returns:
        config (dict): the python dictionary containing configuration data
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        message = f"❌ Configuration file not found at '{CONFIG_PATH}'. \nPlease ensure the file exists or fix path to file."
        log_and_stop(message)
    except json.JSONDecodeError as e:
        message = f"❌ Failed to parse JSON: {e}"
        log_and_stop(message)

    return config


@st.cache_resource
def load_model():
    """Helper function that loads the model's architecture and instantiates a model with its trained weights. Returns agent to cpu in evaluation mode. Optimized using streamlit caching.
    Args:
        N/A
    Returns:
        Agent (torch.nn.Module)
    """
    try:
        model_weights = torch.load(MODEL_WEIGHTS_FULL_PATH, weights_only=True)
        print(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")
    except FileNotFoundError:
        message = f"❌ Model Weights file not found at '{MODEL_WEIGHTS_FULL_PATH}'. \nPlease ensure the file exists."
        log_and_stop(message)

    CONFIG = load_config()
    MODEL_CONFIG = CONFIG.get("model", {})

    try:
        agent = Agent(cfg=MODEL_CONFIG)  # Create agent instance
        agent.load_state_dict(state_dict=model_weights)
    except RuntimeError as e:
        message = f"❌ A runtime error occurred while creating model or loading model weights: {e}"
        log_and_stop(message)
    except FileNotFoundError as e:
        message = f"❌ Model weights file not found: {e}"
        log_and_stop(message)
    except KeyError as e:
        message = f"❌ Missing key in model configuration: {e}"
        log_and_stop(message)

    return agent.eval().to("cpu")


@st.cache_data
def load_feature_scaler():
    """Loads the feature scaler using the global variable. Optimized using streamlit caching.
    Args:
        N/A
    Returns:
        feature_scaler: the loaded scalert object"""
    # Load feature scaler
    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        print(f"✅ Feature Scaler loaded successfully from {FEATURE_SCALER_PATH}")
    except FileNotFoundError:
        message = f"❌ Configuration file not found at '{FEATURE_SCALER_PATH}'. \nPlease ensure the file exists or fix path to file."
        log_and_stop(message)
    return feature_scaler


@st.cache_data
def load_label_scaler():
    """Loads the label scaler using the global variable. Optimized using streamlit caching.
    Args:
        N/A
    Returns:
        label_scaler: the loaded scalert object"""
    # Not used in this implementation
    label_scaler = None

    return label_scaler


def log_and_stop(message: str):
    """Helper function to log to terminal. Shows message to Streamlit UI and exits the program.Args:
        NN/Ane
    Returns:
        N/A
    """

    print(message)  # Console
    st.error(message)  # Streamlit UI
    st.stop()  # Stops Streamlit app

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
    INPUT_METADATA,
    STREAMLIT_VALIDATED,
    MODEL_WEIGHTS_FULL_PATH,
    CONFIG_PATH,
    FEATURE_SCALER_PATH,
)
from .model import Agent


def convert_inputs(**kwargs) -> list:
    """Convert user inputs into a list of features for the model.
    Args:
        **kwargs: Dictionary of user inputs (e.g., {'category': 'entertainment', 'amt': 25.0, ...})
    Returns:
        features: A list of converted features ready for model input.
    """
    features = []  # Create empty list to store all the features

    for feature_name in FEATURE_NAMES:  # Loop through FEATURE_NAMES
        try:
            # Get the value from the kwargs dictionary
            value = kwargs.get(feature_name)

            # Perform validation (using metadata where possible)
            if value is None:
                raise ValueError(f"Missing required input: {feature_name}")

            # --- Mapped Features ---
            if feature_name == "category":
                # Use Specified Mapping for feature
                mapped_value = CATEGORY_MAPPING.get(value, None)
                if mapped_value is not None:
                    if not isinstance(mapped_value, float):
                        raise ValueError(f"{feature_name} must be a float.")
                    features.append(mapped_value)
                else:
                    raise ValueError(f"{feature_name}; value={value}; no mapping.")

            elif feature_name == "gender":
                # Use Specified Mapping for feature
                mapped_value = GENDER_MAPPING.get(value, None)
                if mapped_value is not None:
                    if not isinstance(mapped_value, float):
                        raise ValueError(f"{feature_name} must be a float.")
                    features.append(mapped_value)
                else:
                    raise ValueError(f"{feature_name}; value={value}; no mapping.")

            elif feature_name == "state":
                # Use Specified Mapping for feature
                mapped_value = STATE_MAPPING.get(value, None)
                if mapped_value is not None:
                    if not isinstance(mapped_value, float):
                        raise ValueError(f"{feature_name} must be a float.")
                    features.append(mapped_value)
                else:
                    raise ValueError(f"{feature_name}; value={value}; no mapping.")

            # ... Add logic for other mapped fields here

            # --- Streamlit-Validated Features ---
            elif feature_name in STREAMLIT_VALIDATED:
                # Use INPUT_METADATA for range validation
                meta = INPUT_METADATA.get(feature_name, {})
                min_v = meta.get("min_value")
                max_v = meta.get("max_value")

                if min_v is not None and max_v is not None and not (min_v <= value <= max_v):
                    raise ValueError(f"{feature_name} out of expected range.")
                features.append(float(value))  # Convert to float
            # Default action if not covered by logic above
            else:
                raise ValueError(f"No conversion for {feature_name}")
        except ValueError as e:
            log_and_stop(f"Validation Error for {feature_name}: {e}")

    # Verify final length
    if len(features) != len(FEATURE_NAMES):
        log_and_stop(
            f"Fatal Error: Final feature list length mismatch. Created list size: {len(features)} | Expected list size: {len(FEATURE_NAMES)}"
        )

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
    """Helper function that loads the model's architecture and instantiates a model with its trained weights. Optimized using streamlit caching.
    Args:
        N/A
    Returns:
        Agent (torch.nn.Module): Returns agent to cpu in evaluation mode.
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
        feature_scaler: the loaded scalert object
    """
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
        label_scaler: the loaded scalert object
    """
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

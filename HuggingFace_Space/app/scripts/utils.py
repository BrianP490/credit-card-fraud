"""This module contains utility functions for input conversion and validation."""

import os
import logging
from logging import Logger  # For type hinting
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


def setup_logger(config: dict, propogate: bool = False) -> Logger:
    """Sets up and returns a named logger based on the provided config dictionary. The new logger will have different handlers based on the config.

    Args:
        config (dict): Dictionary containing logging configuration.
        propogate (bool): Whether to allow log messages to propagate to ancestor loggers.
    Returns:
        Logger: Configured logger instance.
    """
    logger_name = config.get("logger_name", "main")
    log_to_file = config.get("log_to_file", True)  # Set whether to log to a logfile or not
    log_file = config.get("log_file", "logs/app.log")  # Get the log file path
    log_lvl = config.get("log_level", "INFO")
    log_level = getattr(logging, log_lvl.upper(), logging.INFO)  # Set fallback if invalid input
    log_mode = config.get("log_mode", "w")  # Set the log file mode
    log_format = config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = config.get("date_format", "%Y-%m-%d %H:%M:%S")
    log_to_console = config.get("log_to_console", True)  # Set whether to log to console or not

    handlers = []  # Initialize the list of logging handlers

    logger = logging.getLogger(logger_name)  # Create logger object with the specified name

    if not log_to_file and not log_to_console:
        # If no handlers are specified by the config
        print(
            f"Warning: No logging handlers configured for {logger_name}.\nVerbose Logging will be disabled.\nIn 'config/config.json', set ['log_to_file': true] or ['log_to_console': true] if you want to change the logging behavior.",
            flush=True,
        )
    else:
        # Create log parent directory if it doesn't exist
        parent_dir = os.path.dirname(log_file)  # Get the parent directory of the log file
        if parent_dir and parent_dir != ".":
            try:
                os.makedirs(name=parent_dir, exist_ok=True)
                print(
                    f"Parent directory '{parent_dir}' used to store the log file.", flush=True
                )  # flush=True to ensure the message is printed immediately
            except OSError as e:
                print(
                    f"Error creating directory '{parent_dir}': {e} INFO: Using default log file 'app.log' instead.",
                    flush=True,
                )
                log_file = "app.log"  # Fall back to a default log file if problem occurs.

        # Remove all old handlers inherrited from the root logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt=log_format, datefmt=date_format
        )  # Create a formatter for the log messages

        if log_to_console:
            console_handler = (
                logging.StreamHandler()
            )  # Initialize sending log messages to the console (stdout)
            console_handler.setFormatter(formatter)  # Set the formatter for the console handler
            handlers.append(console_handler)  # Add the console_handler to the list of handlers
        if log_to_file:
            file_handler = logging.FileHandler(
                filename=log_file, mode=log_mode, encoding="utf-8"
            )  # Initialize sending log messages to a file; Enables emoji use
            file_handler.setFormatter(formatter)  # Set the style for the console handler
            handlers.append(file_handler)  # Add the file_handler to the list of handlers

        # Add the handlers to the logger
        for handler in handlers:
            logger.addHandler(handler)

    logger.setLevel(log_level)  # Set logger minimum log level

    logger.propagate = propogate  # Prevent the log messages from being propagated to the root logger; gets rid of the root logger's default handlers,

    return logger


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
        except Exception as e:  # Catch all other exceptions
            log_and_stop(f"An unexpected fatal error occurred: {e}")

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
    message = ""  # Initialize variable in case of errors
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        # For streamlit to acknowledge the '\n' character as a newline use '  \n'. Streamlit processes strings as Markdown
        message = f"❌ Configuration file not found at '{CONFIG_PATH}'.  \nPlease ensure the file exists or fix path to file."
    except json.JSONDecodeError as e:
        message = f"❌ Failed to parse JSON: {e}."
    except Exception as e:  # Catch all other exceptions
        message = f"An unexpected fatal error occurred: {e}"
    # This block executes ONLY if the 'try' block succeeds (no exceptions)
    else:
        return config

    # **This block executes after try/except/else**
    finally:
        # Check if a 'message' was set by any of the 'except' blocks.
        if message:
            message += "  \nStopping Execution."  # Add the common suffix
            print(message)
            st.error(message)
            st.stop()


@st.cache_resource
def load_model(_logger: Logger):
    """Helper function that loads the model's architecture and instantiates a model with its trained weights. Optimized using streamlit caching.
    Args:
        _logger (Logger): The logger instance to log messages. Use underscore to prevent hashing by Streamlit.
    Returns:
        Agent (torch.nn.Module): Returns agent to cpu in evaluation mode.
    """
    message = ""  # Initialize variable in case of errors
    try:
        model_weights = torch.load(MODEL_WEIGHTS_FULL_PATH, weights_only=True)
        _logger.info(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")
    except FileNotFoundError:
        message = f"❌ Model Weights file not found at '{MODEL_WEIGHTS_FULL_PATH}'.  \nPlease ensure the file exists."
        log_and_stop(message)

    CONFIG = load_config()
    MODEL_CONFIG = CONFIG.get("model", {})

    try:
        agent = Agent(cfg=MODEL_CONFIG)  # Create agent instance
        agent.load_state_dict(state_dict=model_weights)
    except RuntimeError as e:
        message = f"❌ A runtime error occurred while creating model or loading model weights: {e}"
    except FileNotFoundError as e:
        message = f"❌ Model weights file not found: {e}"
    except KeyError as e:
        message = f"❌ Missing key in model configuration: {e}"
    except Exception as e:  # Catch all other exceptions
        message = f"An unexpected fatal error occurred: {e}"
    # Execute if no exception was caught
    else:
        return agent.eval().to("cpu")
    # If exception was thrown continue to the finally block
    finally:
        if message:
            log_and_stop(message)


@st.cache_data
def load_feature_scaler(_logger: Logger):
    """Loads the feature scaler using the global variable. Optimized using streamlit caching.
    Args:
        _logger (Logger): The logger instance to log messages. Use underscore to prevent hashing by Streamlit.
    Returns:
        feature_scaler: the loaded scalert object
    """
    message = ""  # Initialize variable in case of errors
    # Load feature scaler
    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        _logger.info(f"✅ Feature Scaler loaded successfully from {FEATURE_SCALER_PATH}")
    except FileNotFoundError:
        message = f"❌ Scaler file not found at '{FEATURE_SCALER_PATH}'.  \nPlease ensure the file exists or fix path to file."
    except Exception as e:  # Catch all other exceptions
        message = f"An unexpected fatal error occurred: {e}"
    # Execute if no exception was caught
    else:
        return feature_scaler
    # If exception was thrown continue to the finally block
    finally:
        if message:
            log_and_stop(message)


@st.cache_data
def load_label_scaler(_logger: Logger):
    """Loads the label scaler using the global variable. Optimized using streamlit caching.
    Args:
        _logger (Logger): The logger instance to log messages. Use underscore to prevent hashing by Streamlit.
    Returns:
        label_scaler: the loaded scalert object
    """
    # Not used in this implementation
    label_scaler = None

    return label_scaler


def log_and_stop(message: str):
    """Helper function to log relevant messages. Handles message and exits the program.
    Args:
        message (str): The message to log and display
    Returns:
        N/A
    """
    logger_name = load_config()["logging"]["logger_name"]
    logger = logging.getLogger(logger_name)

    message += "  \nStopping Execution."  # Add the common suffix

    logger.info(message, exc_info=False, stack_info=False)  # Console output
    st.error(message)  # Streamlit UI output
    st.stop()  # Stops Streamlit app

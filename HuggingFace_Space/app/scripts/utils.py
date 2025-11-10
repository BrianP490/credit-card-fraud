"""This module contains utility functions for input conversion and validation."""

import streamlit as st

# from scripts import (
#     FEATURE_NAMES,
#     CATEGORY_MAPPING,
#     GENDER_MAPPING,
#     STATE_MAPPING
# )


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
        if category:
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
        if gender:
            features.append(gender)

        # state
        state = args[3]
        if not isinstance(state, str):
            raise ValueError("state must be a string.")
        state = STATE_MAPPING.get(state, None)
        if state:
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
        merch_long = args[2]
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

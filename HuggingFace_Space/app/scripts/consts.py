"""
This module defines constants for file paths and feature names used in the application.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIRECTORY = "model_weights"
MODEL_WEIGHTS_FILE_NAME = "trained-model.pt"
MODEL_WEIGHTS_FULL_PATH = BASE_DIR / MODEL_DIRECTORY / MODEL_WEIGHTS_FILE_NAME

CONFIG_DIRECTORY = "./configs"
CONFIG_FILE_NAME = "config.json"
CONFIG_PATH = BASE_DIR / CONFIG_DIRECTORY / CONFIG_FILE_NAME

SCALER_DIRECTORY = "./Scalers"
FEATURE_SCALER_FILE_NAME = "feature-scaler.joblib"
FEATURE_SCALER_PATH = BASE_DIR / SCALER_DIRECTORY / FEATURE_SCALER_FILE_NAME


# Same feature order names and order as during the model training data set
FEATURE_NAMES = [
    "category",
    "amt",
    "gender",
    "state",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
]

CATEGORY_MAPPING = {
    "entertainment": 0.0,
    "food_dining": 1.0,
    "gas_transport": 2.0,
    "grocery_net": 3.0,
    "grocery_pos": 4.0,
    "health_fitness": 5.0,
    "home": 6.0,
    "kids_pets": 7.0,
    "misc_net": 8.0,
    "misc_pos": 9.0,
    "personal_care": 10.0,
    "shopping_net": 11.0,
    "shopping_pos": 12.0,
    "travel": 13.0,
}

GENDER_MAPPING = {"F": 0.0, "M": 1.0}

STATE_MAPPING = {
    "AK": 0.0,
    "AL": 1.0,
    "AR": 2.0,
    "AZ": 3.0,
    "CA": 4.0,
    "CO": 5.0,
    "CT": 6.0,
    "DC": 7.0,
    "DE": 8.0,
    "FL": 9.0,
    "GA": 10.0,
    "HI": 11.0,
    "IA": 12.0,
    "ID": 13.0,
    "IL": 14.0,
    "IN": 15.0,
    "KS": 16.0,
    "KY": 17.0,
    "LA": 18.0,
    "MA": 19.0,
    "MD": 20.0,
    "ME": 21.0,
    "MI": 22.0,
    "MN": 23.0,
    "MO": 24.0,
    "MS": 25.0,
    "MT": 26.0,
    "NC": 27.0,
    "ND": 28.0,
    "NE": 29.0,
    "NH": 30.0,
    "NJ": 31.0,
    "NM": 32.0,
    "NV": 33.0,
    "NY": 34.0,
    "OH": 35.0,
    "OK": 36.0,
    "OR": 37.0,
    "PA": 38.0,
    "RI": 39.0,
    "SC": 40.0,
    "SD": 41.0,
    "TN": 42.0,
    "TX": 43.0,
    "UT": 44.0,
    "VA": 45.0,
    "VT": 46.0,
    "WA": 47.0,
    "WI": 48.0,
    "WV": 49.0,
    "WY": 50.0,
}

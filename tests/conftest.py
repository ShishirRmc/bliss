"""
Shared fixtures for the Bliss test suite.
"""
import sys
import os

# Make sure bliss/ root is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_customer_df():
    """Minimal customer DataFrame covering every service in the catalog."""
    return pd.DataFrame([
        {"Customer_ID": "C001", "Preferred_Service": "Massage",            "Booking_Frequency": 8,  "Avg_Spending": 120, "Total_Spend": 960},
        {"Customer_ID": "C002", "Preferred_Service": "Facial",             "Booking_Frequency": 5,  "Avg_Spending": 95,  "Total_Spend": 475},
        {"Customer_ID": "C003", "Preferred_Service": "Wellness Package",   "Booking_Frequency": 3,  "Avg_Spending": 179, "Total_Spend": 537},
        {"Customer_ID": "C004", "Preferred_Service": "Manicure",           "Booking_Frequency": 10, "Avg_Spending": 55,  "Total_Spend": 550},
        {"Customer_ID": "C005", "Preferred_Service": "Deep Tissue Massage","Booking_Frequency": 6,  "Avg_Spending": 145, "Total_Spend": 870},
    ])


@pytest.fixture
def fitted_recommender(sample_customer_df):
    """A ContentBasedRecommender already fitted on sample data."""
    from recommender import ContentBasedRecommender
    rec = ContentBasedRecommender()
    rec.fit(sample_customer_df)
    return rec

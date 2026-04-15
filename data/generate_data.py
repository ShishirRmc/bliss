"""
Synthetic customer data generator for the Blys AI assessment.

Generates ~800 customer records with realistic booking patterns,
spending distributions, and sentiment-correlated review texts.

Usage:
    python data/generate_data.py
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ── Service catalog ──────────────────────────────────────────────────────────

SERVICES = [
    "Massage",
    "Facial",
    "Wellness Package",
    "Manicure",
    "Couples Massage",
    "Deep Tissue Massage",
    "Anti-Ageing Facial",
]

SERVICE_PRICES = {
    "Massage": (80, 200),
    "Facial": (60, 150),
    "Wellness Package": (200, 450),
    "Manicure": (40, 90),
    "Couples Massage": (150, 350),
    "Deep Tissue Massage": (100, 220),
    "Anti-Ageing Facial": (90, 180),
}

# ── Review text templates (correlated with sentiment) ────────────────────────

POSITIVE_REVIEWS = [
    "Absolutely loved it! The therapist was amazing.",
    "Best massage I've ever had. Will definitely rebook.",
    "Wonderful experience from start to finish. Highly recommend!",
    "My therapist was incredibly professional and skilled.",
    "Amazing service, felt so relaxed afterwards. 10/10!",
    "Perfect way to unwind after a long week. Thank you Blys!",
    "The booking process was seamless and the service was top-notch.",
    "Exceeded my expectations. The therapist really listened to my needs.",
    "I've tried many services but this was by far the best.",
    "Such a convenient and luxurious experience. Love it!",
    "My facial left my skin glowing. Absolutely fantastic!",
    "The couples massage was the perfect anniversary gift.",
    "Therapist arrived on time and was extremely professional.",
    "Incredible value for the quality of service provided.",
    "This has become my monthly self-care ritual. Can't live without it!",
    "Five stars all around. The experience was flawless.",
    "My go-to for wellness. Never been disappointed once.",
]

NEUTRAL_REVIEWS = [
    "It was okay, nothing special but decent service.",
    "Good experience overall. Average pricing.",
    "The service was fine. Met my basic expectations.",
    "Not bad. The therapist was competent but not exceptional.",
    "Reasonable service for the price. Would consider rebooking.",
    "The booking was easy but the service felt a bit rushed.",
    "Decent quality. Probably would try again.",
    "Standard massage experience. Nothing to complain about.",
    "The therapist was nice enough. Service was adequate.",
    "Okay experience. Not the best I've had but not the worst.",
    "Fair service. The app worked well for booking.",
    "It was a satisfactory experience. Nothing memorable.",
]

NEGATIVE_REVIEWS = [
    "Too expensive for what you get. Disappointing.",
    "The therapist was late and seemed unprepared.",
    "Not worth the price. I've had better at half the cost.",
    "Booking process was confusing and the service was mediocre.",
    "I was not impressed. Expected much more for the price.",
    "The experience did not match the website description at all.",
    "Therapist cancelled last minute. Very frustrating.",
    "Poor communication and the quality was below average.",
    "Would not recommend. The whole experience felt unprofessional.",
    "Overpriced and underwhelming. Won't be coming back.",
    "My skin reacted badly to the products used. No follow-up from support.",
    "The service was rushed and I felt like just another number.",
]

# ── Customer archetypes ──────────────────────────────────────────────────────

ARCHETYPES = {
    "loyalist": {
        "weight": 0.20,
        "frequency": (8, 20),
        "spending": (150, 400),
        "recency_days": (1, 30),
        "preferred_services": ["Massage", "Wellness Package", "Couples Massage", "Deep Tissue Massage"],
        "sentiment_dist": (0.85, 0.10, 0.05),  # pos, neu, neg
    },
    "occasional": {
        "weight": 0.30,
        "frequency": (2, 5),
        "spending": (50, 120),
        "recency_days": (15, 90),
        "preferred_services": ["Massage", "Facial", "Manicure", "Anti-Ageing Facial"],
        "sentiment_dist": (0.40, 0.40, 0.20),
    },
    "lapsed": {
        "weight": 0.20,
        "frequency": (5, 15),
        "spending": (120, 350),
        "recency_days": (90, 365),
        "preferred_services": ["Massage", "Wellness Package", "Facial", "Deep Tissue Massage"],
        "sentiment_dist": (0.30, 0.30, 0.40),
    },
    "new_explorer": {
        "weight": 0.20,
        "frequency": (1, 3),
        "spending": (60, 180),
        "recency_days": (1, 20),
        "preferred_services": SERVICES,
        "sentiment_dist": (0.60, 0.30, 0.10),
    },
    "churned": {
        "weight": 0.10,
        "frequency": (1, 3),
        "spending": (40, 100),
        "recency_days": (180, 500),
        "preferred_services": ["Massage", "Facial", "Anti-Ageing Facial"],
        "sentiment_dist": (0.15, 0.25, 0.60),
    },
}


def _pick_review(sentiment_dist: tuple) -> str:
    """Pick a review text based on sentiment probability distribution."""
    category = np.random.choice(
        ["positive", "neutral", "negative"], p=sentiment_dist
    )
    if category == "positive":
        return random.choice(POSITIVE_REVIEWS)
    elif category == "neutral":
        return random.choice(NEUTRAL_REVIEWS)
    else:
        return random.choice(NEGATIVE_REVIEWS)


def generate_customers(n: int = 800) -> pd.DataFrame:
    """Generate n synthetic customer records."""
    records = []
    customer_id = 1001
    reference_date = datetime(2024, 3, 15)

    for archetype_name, config in ARCHETYPES.items():
        count = int(n * config["weight"])

        for _ in range(count):
            freq = np.random.randint(*config["frequency"])
            avg_spend = round(
                np.random.uniform(*config["spending"])
                + np.random.normal(0, 15),
                2,
            )
            avg_spend = max(avg_spend, 30)  # floor

            days_since = np.random.randint(*config["recency_days"])
            last_activity = reference_date - timedelta(days=int(days_since))

            total_spend = round(avg_spend * freq * np.random.uniform(0.8, 1.2), 2)

            preferred_service = random.choice(config["preferred_services"])
            review_text = _pick_review(config["sentiment_dist"])

            records.append(
                {
                    "Customer_ID": customer_id,
                    "Booking_Frequency": freq,
                    "Avg_Spending": avg_spend,
                    "Preferred_Service": preferred_service,
                    "Review_Text": review_text,
                    "Last_Activity": last_activity.strftime("%Y-%m-%d"),
                    "Days_Since_Last_Booking": int(days_since),
                    "Total_Spend": total_spend,
                }
            )
            customer_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    df = generate_customers(800)
    out_path = os.path.join(os.path.dirname(__file__), "customer_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} customer records → {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nService distribution:\n{df['Preferred_Service'].value_counts()}")
    print(f"\nSpending stats:\n{df['Avg_Spending'].describe()}")


if __name__ == "__main__":
    main()

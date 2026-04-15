# Customer Behaviour Analysis — Blys

## Dataset
- **Records:** 800 synthetic customers
- **Fields:** Customer_ID, Booking_Frequency, Avg_Spending, Preferred_Service, Review_Text, Last_Activity, Days_Since_Last_Booking, Total_Spend
- **Generation:** Archetype-based synthetic data with realistic booking patterns and sentiment-correlated reviews

## Preprocessing
- Missing values filled with column median
- Sentiment extracted from Review_Text using TextBlob (polarity: -1 to +1)
- Booking_Frequency and Avg_Spending normalised with StandardScaler

## Customer Segments (K-Means, k=4)

| Segment | Count | Avg Frequency | Avg Spending | Avg Sentiment | Avg Days Inactive |
|---|---|---|---|---|---|
| High-Value Loyalists | 99 | 9.06 | $308.23 | 0.3 | 134.59 days |
| Lapsed High-Spenders | 117 | 10.94 | $187.8 | 0.08 | 167.25 days |
| New Explorers | 487 | 2.26 | $95.36 | 0.23 | 87.4 days |
| Price-Sensitive Occasionals | 97 | 15.69 | $298.74 | 0.32 | 55.49 days |

## Business Recommendations

### High-Value Loyalists
- Highest LTV segment. Assign top-rated therapists to this group.
- Offer a loyalty programme (e.g. every 10th booking free) to reinforce retention.

### Lapsed High-Spenders
- Highest churn risk despite historically high spend.
- Re-engagement campaign: personalised email with 15% discount on their preferred service.
- Target within 90-day inactivity window before they become fully churned.

### Price-Sensitive Occasionals
- Respond well to promotions. Bundle deals (e.g. Massage + Manicure) can increase basket size.
- Upsell to Wellness Package with a first-time discount.

### New Explorers
- Recent first-timers with positive sentiment — high conversion potential.
- Follow-up within 7 days with a personalised recommendation based on their first service.
- Onboarding email series to introduce the full service catalogue.

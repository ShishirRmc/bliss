# Blys Customer Behavior Analysis & Insights Report

## 1. Executive Summary
This report analyzes Blys customer interaction patterns utilizing K-Means clustering and Natural Language Processing (NLP) sentiment scoring. The data captures historical bounds of `Booking_Frequency`, `Avg_Spending`, and textual analysis of `Review_Text`.

## 2. Methodology
- **Data Preprocessing**: Missing `Booking_Frequency` and `Avg_Spending` values were imputed using column medians, and numeric features were normalized utilizing statistical standard scaling. 
- **Sentiment Analysis**: NLP processing via the `textblob` toolkit was applied to extract a `Review_Sentiment` polarity score between -1.0 (highly negative) and 1.0 (highly positive).
- **Segmentation Strategy**: A `K-Means (n=3)` clustering algorithm was selected to segment broad behavior patterns programmatically utilizing normalized spend characteristics and sentiment satisfaction bounds.

## 3. Customer Segments
Based on algorithmic grouping, three distinct cohorts emerged from the underlying variance maps:
1. **VIP / High-Value Customers (Cluster 1)**: Exhibit highest average spending (`Avg_Spending` bounds upper quartile) and frequent bookings. Their sentiment rating is consistently optimal (> 0.5 polarity).
2. **Average / Active Customers (Cluster 2)**: Mainstream customers utilizing services semi-regularly with moderate spending margins and generally positive/neutral reviews.
3. **At-Risk / Churn Customers (Cluster 3)**: Identified primarily through negative sentiment polarity in `Review_Text` (< 0.0 score) alongside stalling or lower booking frequencies.

## 4. Key Insights & Engagement Recommendations

### Retention Strategies for High-Value VIPs
- **Automated Rebooking Priority**: Allow frictionless "One-Tap Rebooking" through the platform focusing on their explicitly mapped `Preferred_Service`.
- **Value-Add Upgrades**: VIPs frequently select broad packages over individual variants. Trigger proactive retention sequences by providing complimentary aroma/hot-stone additions to bookings exceeding standard lengths.

### Target Engagement for At-Risk Churn Group
- **Immediate Recovery Pipelines**: For any user triggering a sub-zero sentiment score, alert customer-support nodes automatically.
- **Incentive Outreach Campaigns**: Use the downstream Personalization Recommendation engine to locate a high-confidence alternative service map, offering them an exclusive discount to trial it, demonstrating high-quality service recovery.

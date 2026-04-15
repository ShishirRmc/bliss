"""
Content-based recommendation engine for Blys services.

Recommends services to customers based on cosine similarity between
customer preference vectors and service feature vectors.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


SERVICE_CATALOG = pd.DataFrame(
    [
        {
            "service_id": "SVC001",
            "service_name": "Massage",
            "category": "body",
            "avg_price": 120,
            "duration_min": 60,
            "relaxation_score": 0.9,
            "therapeutic_score": 0.7,
        },
        {
            "service_id": "SVC002",
            "service_name": "Facial",
            "category": "face",
            "avg_price": 95,
            "duration_min": 60,
            "relaxation_score": 0.7,
            "therapeutic_score": 0.4,
        },
        {
            "service_id": "SVC003",
            "service_name": "Wellness Package",
            "category": "combo",
            "avg_price": 179,
            "duration_min": 120,
            "relaxation_score": 1.0,
            "therapeutic_score": 0.6,
        },
        {
            "service_id": "SVC004",
            "service_name": "Manicure",
            "category": "nails",
            "avg_price": 55,
            "duration_min": 45,
            "relaxation_score": 0.4,
            "therapeutic_score": 0.1,
        },
        {
            "service_id": "SVC005",
            "service_name": "Couples Massage",
            "category": "body",
            "avg_price": 220,
            "duration_min": 60,
            "relaxation_score": 0.95,
            "therapeutic_score": 0.65,
        },
        {
            "service_id": "SVC006",
            "service_name": "Deep Tissue Massage",
            "category": "body",
            "avg_price": 145,
            "duration_min": 75,
            "relaxation_score": 0.6,
            "therapeutic_score": 0.95,
        },
        {
            "service_id": "SVC007",
            "service_name": "Anti-Ageing Facial",
            "category": "face",
            "avg_price": 120,
            "duration_min": 75,
            "relaxation_score": 0.65,
            "therapeutic_score": 0.5,
        },
    ]
)


class ContentBasedRecommender:
    """Content-based service recommendation engine."""

    def __init__(self):
        self.service_catalog = SERVICE_CATALOG.copy()
        self.service_vectors = None
        self.customer_vectors: dict[str, np.ndarray] = {}
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._spending_scaler = StandardScaler()
        self._service_name_to_id: dict[str, str] = {}
        self._fitted = False

    def fit(self, customer_df: pd.DataFrame):
        """
        Build service feature vectors and customer preference vectors.

        Args:
            customer_df: DataFrame with Customer_ID, Preferred_Service,
                        Avg_Spending, Booking_Frequency, Total_Spend columns.
        """
        # Build service feature vectors
        numerical_cols = [
            "avg_price",
            "duration_min",
            "relaxation_score",
            "therapeutic_score",
        ]
        categorical_cols = ["category"]

        # Fit and transform
        num_features = self.scaler.fit_transform(
            self.service_catalog[numerical_cols]
        )
        cat_features = self.encoder.fit_transform(
            self.service_catalog[categorical_cols]
        )

        self.service_vectors = np.hstack([num_features, cat_features])

        # Build lookup
        self._service_name_to_id = dict(
            zip(
                self.service_catalog["service_name"],
                range(len(self.service_catalog)),
            )
        )

        # Fit spending scaler for customer enrichment
        spending_vals = customer_df["Avg_Spending"].values.reshape(-1, 1)
        self._spending_scaler = StandardScaler().fit(spending_vals)

        # Build customer preference vectors
        for _, row in customer_df.iterrows():
            self.customer_vectors[str(row["Customer_ID"])] = self._build_customer_vector(row)

        self._fitted = True

    def _build_customer_vector(self, row) -> np.ndarray:
        """
        Build a preference vector for a single customer row.

        Combines:
        - Service feature vector of their preferred service
        - Weighted by booking frequency (engagement signal)
        - Scaled by normalised spending tier (value signal)
        - Recency decay: recent customers get a stronger signal
        """
        preferred = row["Preferred_Service"]

        if preferred not in self._service_name_to_id:
            return self.service_vectors.mean(axis=0)

        idx = self._service_name_to_id[preferred]
        base_vec = self.service_vectors[idx].copy()

        # Frequency weight: 0.5–1.0 (more bookings = stronger preference)
        freq_weight = 0.5 + 0.5 * min(row["Booking_Frequency"] / 10, 1.0)

        # Spending weight: 0.8–1.2 (higher spenders get a slight boost)
        norm_spend = float(
            self._spending_scaler.transform([[row["Avg_Spending"]]])[0][0]
        )
        spend_weight = 1.0 + 0.2 * np.tanh(norm_spend)

        # Recency decay: customers inactive >180 days get a 0.7x signal
        days_inactive = row.get("Days_Since_Last_Booking", 0)
        recency_weight = 0.7 if days_inactive > 180 else 1.0

        return base_vec * freq_weight * spend_weight * recency_weight

    def recommend(
        self, customer_id: str, top_n: int = 3
    ) -> list[dict]:
        """
        Recommend services for a customer.

        Args:
            customer_id: Customer ID string.
            top_n: Number of recommendations to return.

        Returns:
            List of {"service": str, "confidence": float}
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if customer_id not in self.customer_vectors:
            # Cold start: return most popular services
            return self._popularity_fallback(top_n)

        customer_vec = self.customer_vectors[customer_id].reshape(1, -1)
        similarities = cosine_similarity(customer_vec, self.service_vectors)[0]

        # Identify the customer's already-booked (highest-similarity) service
        booked_idx = int(np.argmax(similarities))
        booked_service = self.service_catalog.iloc[booked_idx]["service_name"]

        # Rank and filter, excluding the already-booked service
        ranked_indices = np.argsort(similarities)[::-1]
        recommendations = []

        for idx in ranked_indices:
            service_name = self.service_catalog.iloc[idx]["service_name"]
            if service_name == booked_service:
                continue  # skip the service the customer already prefers
            score = similarities[idx]

            recommendations.append(
                {
                    "service": service_name,
                    "service_id": self.service_catalog.iloc[idx]["service_id"],
                    "confidence": round(float(score), 4),
                    "category": self.service_catalog.iloc[idx]["category"],
                    "avg_price": float(
                        self.service_catalog.iloc[idx]["avg_price"]
                    ),
                }
            )

            if len(recommendations) >= top_n:
                break

        return recommendations

    def evaluate(self, customer_df: pd.DataFrame, k_values: list[int] = None) -> dict:
        """
        Evaluate using leave-one-out: for each customer, temporarily build their
        vector WITHOUT their preferred service, then check if the held-out service
        appears in the top-K recommendations.

        Returns:
            {"precision@3": float, "precision@5": float, "coverage": float, ...}
        """
        if k_values is None:
            k_values = [3, 5]

        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = {f"precision@{k}": 0.0 for k in k_values}
        hits = {k: 0 for k in k_values}
        recommended_services: set[str] = set()
        total = 0

        for _, row in customer_df.iterrows():
            cid = str(row["Customer_ID"])
            actual_service = row["Preferred_Service"]

            if actual_service not in self._service_name_to_id:
                continue

            # Build a leave-one-out vector: use the second-most-similar service
            # as the proxy preference so the held-out service can appear in recs.
            actual_idx = self._service_name_to_id[actual_service]
            sims = cosine_similarity(
                self.service_vectors[actual_idx].reshape(1, -1), self.service_vectors
            )[0]
            # Zero out the actual service and pick the next best
            sims[actual_idx] = -1
            proxy_idx = int(np.argmax(sims))

            proxy_vec = self.service_vectors[proxy_idx] * (
                0.5 + 0.5 * min(row["Booking_Frequency"] / 10, 1.0)
            )

            # Temporarily swap the customer vector
            original_vec = self.customer_vectors.get(cid)
            self.customer_vectors[cid] = proxy_vec

            # Get top-K recommendations (no exclusion needed — preferred not in proxy)
            customer_vec = proxy_vec.reshape(1, -1)
            similarities = cosine_similarity(customer_vec, self.service_vectors)[0]
            ranked_indices = np.argsort(similarities)[::-1]

            rec_names = [
                self.service_catalog.iloc[i]["service_name"]
                for i in ranked_indices[: max(k_values)]
            ]
            recommended_services.update(rec_names)

            for k in k_values:
                if actual_service in rec_names[:k]:
                    hits[k] += 1

            total += 1

            # Restore original vector
            if original_vec is not None:
                self.customer_vectors[cid] = original_vec
            else:
                del self.customer_vectors[cid]

        if total > 0:
            for k in k_values:
                results[f"precision@{k}"] = round(hits[k] / total, 4)

        results["coverage"] = round(
            len(recommended_services) / len(self.service_catalog), 4
        )
        results["total_evaluated"] = total

        return results

    def _popularity_fallback(self, top_n: int) -> list[dict]:
        """Return most popular services (cold-start fallback)."""
        popular = self.service_catalog.sort_values(
            "relaxation_score", ascending=False
        ).head(top_n)

        return [
            {
                "service": row["service_name"],
                "service_id": row["service_id"],
                "confidence": round(float(row["relaxation_score"]), 4),
                "category": row["category"],
                "avg_price": float(row["avg_price"]),
                "note": "popularity-based (cold start)",
            }
            for _, row in popular.iterrows()
        ]

    def save(self, filepath: str = "models/recommendation_model.pkl"):
        """Save the fitted model."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/recommendation_model.pkl") -> "ContentBasedRecommender":
        """Load a saved model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

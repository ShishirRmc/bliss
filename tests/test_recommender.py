"""
Tests for ContentBasedRecommender.

Covers:
- fit() builds vectors correctly
- recommend() excludes the booked service (bug fix)
- recommend() returns exactly top_n results
- cold-start fallback for unknown customers
- evaluate() returns valid precision/coverage metrics
- save/load round-trip
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from recommender import ContentBasedRecommender, SERVICE_CATALOG


# ── Helpers ──────────────────────────────────────────────────────────────────

ALL_SERVICES = SERVICE_CATALOG["service_name"].tolist()


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fitted_flag_set(self, fitted_recommender):
        assert fitted_recommender._fitted is True

    def test_service_vectors_shape(self, fitted_recommender):
        # 7 services × (4 numerical + N one-hot categories)
        assert fitted_recommender.service_vectors.shape[0] == len(SERVICE_CATALOG)

    def test_customer_vectors_populated(self, fitted_recommender, sample_customer_df):
        for cid in sample_customer_df["Customer_ID"].astype(str):
            assert cid in fitted_recommender.customer_vectors

    def test_unfitted_raises(self):
        rec = ContentBasedRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            rec.recommend("C001")

    def test_unknown_preferred_service_uses_mean(self, sample_customer_df):
        """Customers with an unrecognised preferred service get the mean vector."""
        df = sample_customer_df.copy()
        df.loc[0, "Preferred_Service"] = "NonExistentService"
        rec = ContentBasedRecommender()
        rec.fit(df)
        expected_mean = rec.service_vectors.mean(axis=0)
        np.testing.assert_array_almost_equal(
            rec.customer_vectors["C001"], expected_mean
        )


# ── recommend() ───────────────────────────────────────────────────────────────

class TestRecommend:
    @pytest.mark.parametrize("customer_id,preferred", [
        ("C001", "Massage"),
        ("C002", "Facial"),
        ("C003", "Wellness Package"),
        ("C004", "Manicure"),
        ("C005", "Deep Tissue Massage"),
    ])
    def test_booked_service_excluded(self, fitted_recommender, customer_id, preferred):
        """The customer's preferred (booked) service must NOT appear in results."""
        recs = fitted_recommender.recommend(customer_id, top_n=3)
        names = [r["service"] for r in recs]
        assert preferred not in names, (
            f"Booked service '{preferred}' should be excluded but got: {names}"
        )

    @pytest.mark.parametrize("top_n", [1, 2, 3, 5])
    def test_returns_exactly_top_n(self, fitted_recommender, top_n):
        recs = fitted_recommender.recommend("C001", top_n=top_n)
        assert len(recs) == top_n

    def test_result_keys_present(self, fitted_recommender):
        recs = fitted_recommender.recommend("C001", top_n=2)
        for r in recs:
            assert "service" in r
            assert "service_id" in r
            assert "confidence" in r
            assert "category" in r
            assert "avg_price" in r

    def test_confidence_scores_in_range(self, fitted_recommender):
        recs = fitted_recommender.recommend("C001", top_n=3)
        for r in recs:
            assert -1.0 <= r["confidence"] <= 1.0

    def test_results_are_valid_catalog_services(self, fitted_recommender):
        recs = fitted_recommender.recommend("C001", top_n=3)
        for r in recs:
            assert r["service"] in ALL_SERVICES

    def test_cold_start_unknown_customer(self, fitted_recommender):
        """Unknown customer ID triggers popularity fallback."""
        recs = fitted_recommender.recommend("UNKNOWN_999", top_n=3)
        assert len(recs) == 3
        for r in recs:
            assert "note" in r
            assert r["note"] == "popularity-based (cold start)"

    def test_cold_start_returns_valid_services(self, fitted_recommender):
        recs = fitted_recommender.recommend("GHOST", top_n=2)
        for r in recs:
            assert r["service"] in ALL_SERVICES


# ── evaluate() ────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_expected_keys(self, fitted_recommender, sample_customer_df):
        results = fitted_recommender.evaluate(sample_customer_df)
        assert "precision@3" in results
        assert "precision@5" in results
        assert "coverage" in results
        assert "total_evaluated" in results

    def test_precision_in_range(self, fitted_recommender, sample_customer_df):
        results = fitted_recommender.evaluate(sample_customer_df)
        assert 0.0 <= results["precision@3"] <= 1.0
        assert 0.0 <= results["precision@5"] <= 1.0

    def test_coverage_in_range(self, fitted_recommender, sample_customer_df):
        results = fitted_recommender.evaluate(sample_customer_df)
        assert 0.0 <= results["coverage"] <= 1.0

    def test_total_evaluated_matches_known_customers(self, fitted_recommender, sample_customer_df):
        results = fitted_recommender.evaluate(sample_customer_df)
        assert results["total_evaluated"] == len(sample_customer_df)

    def test_custom_k_values(self, fitted_recommender, sample_customer_df):
        results = fitted_recommender.evaluate(sample_customer_df, k_values=[1, 10])
        assert "precision@1" in results
        assert "precision@10" in results


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_round_trip(self, fitted_recommender):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_recommender.save(path)
            loaded = ContentBasedRecommender.load(path)
            assert loaded._fitted is True
            original = fitted_recommender.recommend("C001", top_n=3)
            restored = loaded.recommend("C001", top_n=3)
            assert [r["service"] for r in original] == [r["service"] for r in restored]
        finally:
            os.unlink(path)

"""Tests for the token cost calculator tool."""

import re
import pytest
from personal_assistant.tools.token_cost_calculator import (
    calculate_interaction_cost,
    estimate_batch_cost,
    PRICING,
)


class TestCalculateInteractionCost:
    def test_known_model_returns_cost_string(self):
        result = calculate_interaction_cost("gemini-2.0-flash-001", 1000, 500)
        assert "Token Cost Breakdown" in result
        assert "gemini-2.0-flash-001" in result

    def test_zero_tokens_returns_zero_cost(self):
        result = calculate_interaction_cost("gemini-2.0-flash-001", 0, 0)
        assert "$0.000000" in result

    def test_unknown_model_returns_error(self):
        result = calculate_interaction_cost("not-a-real-model", 100, 100)
        assert "not found" in result.lower()
        assert "not-a-real-model" in result

    def test_input_and_output_costs_are_itemised(self):
        result = calculate_interaction_cost("gemini-1.5-pro", 1000, 1000)
        assert "Input tokens" in result
        assert "Output tokens" in result
        assert "Total cost" in result

    def test_all_pricing_models_are_computable(self):
        for model in PRICING:
            result = calculate_interaction_cost(model, 500, 200)
            assert "Total cost" in result, f"Failed for model: {model}"


class TestEstimateBatchCost:
    def test_known_model_returns_estimate(self):
        result = estimate_batch_cost("gemini-2.0-flash-001", 100, 500, 200)
        assert "Batch Cost Estimate" in result
        assert "Interactions: 100" in result

    def test_unknown_model_returns_error(self):
        result = estimate_batch_cost("not-a-real-model", 10, 100, 50)
        assert "not found" in result.lower()

    def test_single_interaction_matches_calculator(self):
        """Batch cost for 1 interaction should match individual calculator."""
        batch = estimate_batch_cost("gemini-2.0-flash-001", 1, 1000, 500)
        single = calculate_interaction_cost("gemini-2.0-flash-001", 1000, 500)
        # Both should have the same total; extract dollar values and compare
        batch_total = float(re.search(r"\$(\d+\.\d+)", batch.split("total cost")[-1], re.IGNORECASE).group(1))
        single_total = float(re.search(r"\$(\d+\.\d+)", single.split("Total cost")[-1], re.IGNORECASE).group(1))
        # Batch output is rounded to 2 dp, single to 6 dp — allow 1 cent tolerance
        assert abs(batch_total - single_total) < 0.01

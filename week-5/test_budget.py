
import unittest

from budget import calculate_cost, check_rate_limits


class TestBudget(unittest.TestCase):

    def setUp(self):
        self.config = {
            "input_price": 0.30,
            "output_price": 1.20,
            "requests_per_day": 2000,
            "avg_input_tokens": 1200,
            "avg_output_tokens": 300,
            "peak_rpm": 30,
            "rpm_limit": 60,
            "tpm_limit": 100000,
            "monthly_budget": 50,
            "daily_budget": 2,
            "safety_margin": 20
        }

    def test_calculate_cost(self):
        cost = calculate_cost(1000000, 1000000, self.config)
        self.assertAlmostEqual(cost, 1.50)

    def test_rate_limits_within_limit(self):
        result = check_rate_limits(self.config)
        self.assertTrue(result["rpm_ok"])
        self.assertTrue(result["tpm_ok"])

    def test_rate_limits_exceeded(self):
        self.config["peak_rpm"] = 100
        result = check_rate_limits(self.config)
        self.assertFalse(result["rpm_ok"])
        self.assertFalse(result["tpm_ok"])


if __name__ == "__main__":
    unittest.main()


import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_config():
    return {
        "input_price": float(os.getenv("INPUT_PRICE_PER_MILLION")),
        "output_price": float(os.getenv("OUTPUT_PRICE_PER_MILLION")),
        "requests_per_day": int(os.getenv("REQUESTS_PER_DAY")),
        "avg_input_tokens": int(os.getenv("AVG_INPUT_TOKENS")),
        "avg_output_tokens": int(os.getenv("AVG_OUTPUT_TOKENS")),
        "peak_rpm": int(os.getenv("PEAK_REQUESTS_PER_MINUTE")),
        "rpm_limit": int(os.getenv("RPM_LIMIT")),
        "tpm_limit": int(os.getenv("TPM_LIMIT")),
        "monthly_budget": float(os.getenv("MONTHLY_BUDGET_USD")),
        "daily_budget": float(os.getenv("DAILY_BUDGET_USD")),
        "safety_margin": float(os.getenv("SAFETY_MARGIN_PERCENT"))
    }


def calculate_cost(input_tokens, output_tokens, config):
    input_cost = (
        input_tokens / 1_000_000
    ) * config["input_price"]

    output_cost = (
        output_tokens / 1_000_000
    ) * config["output_price"]

    return input_cost + output_cost


def calculate_daily_usage(config):
    daily_input_tokens = (
        config["requests_per_day"] * config["avg_input_tokens"]
    )

    daily_output_tokens = (
        config["requests_per_day"] * config["avg_output_tokens"]
    )

    daily_cost = calculate_cost(
        daily_input_tokens,
        daily_output_tokens,
        config
    )

    return daily_input_tokens, daily_output_tokens, daily_cost


def check_rate_limits(config):
    tokens_per_request = (
        config["avg_input_tokens"] + config["avg_output_tokens"]
    )

    estimated_tpm = config["peak_rpm"] * tokens_per_request

    return {
        "rpm_ok": config["peak_rpm"] <= config["rpm_limit"],
        "tpm_ok": estimated_tpm <= config["tpm_limit"],
        "estimated_tpm": estimated_tpm
    }


def main():
    config = get_config()

    input_tokens, output_tokens, daily_cost = calculate_daily_usage(config)

    monthly_cost = daily_cost * 30

    safe_daily_budget = (
        config["daily_budget"]
        * (1 - config["safety_margin"] / 100)
    )

    rate_status = check_rate_limits(config)

    print("\n========== TOKEN ECONOMICS REPORT ==========")

    print("\nDAILY TOKEN USAGE")
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)
    print("Total tokens:", input_tokens + output_tokens)

    print("\nCOST ESTIMATION")
    print("Daily cost: $", round(daily_cost, 4))
    print("Monthly cost: $", round(monthly_cost, 2))

    print("\nBUDGET CHECK")
    print("Daily budget: $", config["daily_budget"])
    print("Safe daily budget after margin: $", round(safe_daily_budget, 2))
    print("Monthly budget: $", config["monthly_budget"])

    print("Daily budget status:",
          "WITHIN BUDGET" if daily_cost <= safe_daily_budget else "OVER BUDGET")

    print("Monthly budget status:",
          "WITHIN BUDGET" if monthly_cost <= config["monthly_budget"] else "OVER BUDGET")

    print("\nRATE LIMIT CHECK")
    print("Peak requests per minute:", config["peak_rpm"])
    print("RPM limit:", config["rpm_limit"])
    print("Estimated tokens per minute:", rate_status["estimated_tpm"])
    print("TPM limit:", config["tpm_limit"])

    print("RPM status:", "OK" if rate_status["rpm_ok"] else "EXCEEDED")
    print("TPM status:", "OK" if rate_status["tpm_ok"] else "EXCEEDED")

    print("\n============================================")


if __name__ == "__main__":
    main()

"""Token cost calculator for Vertex AI models."""

from google.adk.tools import FunctionTool

# Pricing per 1,000 tokens (as of Dec 2025 for Vertex AI models)
# Update these rates based on current Vertex AI pricing
PRICING = {
    "gemini-2.0-flash-001": {
        "input": 0.075,      # $0.075 per 1K input tokens
        "output": 0.30,      # $0.30 per 1K output tokens
    },
    "gemini-2.0-flash-live-001": {
        "input": 0.075,      # Same as standard for now
        "output": 0.30,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-1.5-flash-002": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-1.5-pro": {
        "input": 1.25,       # $1.25 per 1K input tokens
        "output": 5.00,      # $5.00 per 1K output tokens
    },
}


def calculate_interaction_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> str:
    """Calculate cost of a single model interaction based on tokens.
    
    Args:
        model: Model name (e.g., "gemini-2.0-flash-001")
        input_tokens: Number of input tokens sent to the model
        output_tokens: Number of output tokens received from the model
    
    Returns:
        Formatted cost breakdown string
    """
    if model not in PRICING:
        available = ", ".join(PRICING.keys())
        return (
            f"Model '{model}' not found in pricing table.\n"
            f"Available models: {available}"
        )
    
    rates = PRICING[model]
    input_cost = (input_tokens / 1000) * rates["input"]
    output_cost = (output_tokens / 1000) * rates["output"]
    total_cost = input_cost + output_cost
    
    return (
        f"**Token Cost Breakdown for {model}**\n"
        f"- Input tokens: {input_tokens:,} → ${input_cost:.6f}\n"
        f"- Output tokens: {output_tokens:,} → ${output_cost:.6f}\n"
        f"- **Total cost: ${total_cost:.6f}**"
    )


def estimate_batch_cost(
    model: str,
    num_interactions: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
) -> str:
    """Estimate total cost for multiple interactions.
    
    Args:
        model: Model name
        num_interactions: Number of interactions
        avg_input_tokens: Average input tokens per interaction
        avg_output_tokens: Average output tokens per interaction
    
    Returns:
        Estimated cost breakdown string
    """
    if model not in PRICING:
        available = ", ".join(PRICING.keys())
        return (
            f"Model '{model}' not found in pricing table.\n"
            f"Available models: {available}"
        )
    
    rates = PRICING[model]
    total_input_tokens = avg_input_tokens * num_interactions
    total_output_tokens = avg_output_tokens * num_interactions
    
    input_cost = (total_input_tokens / 1000) * rates["input"]
    output_cost = (total_output_tokens / 1000) * rates["output"]
    total_cost = input_cost + output_cost
    
    return (
        f"**Batch Cost Estimate for {model}**\n"
        f"- Interactions: {num_interactions:,}\n"
        f"- Total input tokens: {total_input_tokens:,} → ${input_cost:.2f}\n"
        f"- Total output tokens: {total_output_tokens:,} → ${output_cost:.2f}\n"
        f"- **Estimated total cost: ${total_cost:.2f}**"
    )


# Create tool instances
token_cost_calculator_tool = FunctionTool(calculate_interaction_cost)
batch_cost_estimator_tool = FunctionTool(estimate_batch_cost)

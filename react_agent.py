import json
import re
import requests

from ml_tools import (
    load_dataset_summary,
    train_sklearn_model,
    train_pytorch_mlp,
)


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


# ============================================================
# CALL OLLAMA
# ============================================================

def call_llm(prompt: str) -> str:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "").strip()


# ============================================================
# PARSE LLM RESPONSE
# ============================================================

def parse_response(response: str):

    final_match = re.search(
        r"Final Answer:\s*(.*)",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if final_match:
        return {
            "type": "final",
            "answer": final_match.group(1).strip()
        }

    action_match = re.search(
        r"Action:\s*([a-zA-Z0-9_]+)",
        response,
        re.IGNORECASE
    )

    if action_match:

        action = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*([^\r\n]+)",
            response,
            re.IGNORECASE
        )

        if input_match:
            return {
                "type": "action",
                "action": action,
                "action_input": input_match.group(1).strip()
            }

    # Tolerant parser for smaller local models

    tool_names = "|".join(
        re.escape(name)
        for name in TOOLS.keys()
    )

    loose_action_match = re.search(
        rf"\b({tool_names})\b",
        response,
        re.IGNORECASE
    )

    if loose_action_match:

        action = loose_action_match.group(1).strip()

        if action == "load_dataset_summary":

            dataset_match = re.search(
                r"\b(iris|wine|breast_cancer)\b",
                response,
                re.IGNORECASE
            )

            if dataset_match:
                return {
                    "type": "action",
                    "action": action,
                    "action_input": dataset_match.group(1)
                }

        if action == "train_sklearn_model":

            dataset_match = re.search(
                r"(?:dataset_name|dataset)\s*:\s*([a-zA-Z_]+)",
                response,
                re.IGNORECASE
            )

            model_match = re.search(
                r"(?:model|model_name|model_type)\s*:\s*([a-zA-Z_]+)",
                response,
                re.IGNORECASE
            )

            if dataset_match and model_match:

                return {
                    "type": "action",
                    "action": action,
                    "action_input": (
                        f"{dataset_match.group(1)}, "
                        f"{model_match.group(1)}"
                    )
                }

        if action == "train_pytorch_mlp":

            dataset_match = re.search(
                r"\b(iris|wine|breast_cancer)\b",
                response,
                re.IGNORECASE
            )

            if dataset_match:
                return {
                    "type": "action",
                    "action": action,
                    "action_input": dataset_match.group(1)
                }

    return {
        "type": "error",
        "message": "No valid Action found."
    }


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(action: str, action_input: str):

    try:

        if action not in TOOLS:
            return "ERROR: Invalid tool."

        if action == "load_dataset_summary":

            dataset_name = action_input.strip()

            result = TOOLS[action](dataset_name)

            if isinstance(result, dict):
                return json.dumps(
                    result,
                    ensure_ascii=False
                )

            return str(result)

        if action == "train_sklearn_model":

            parts = [
                part.strip()
                for part in action_input.split(",", 1)
            ]

            if len(parts) != 2:
                return (
                    "ERROR: train_sklearn_model requires "
                    "dataset and model."
                )

            dataset_name = parts[0]
            model_name = parts[1]

            result = TOOLS[action](
                dataset_name,
                model_name
            )

            if isinstance(result, dict):
                return json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )

            return str(result)

        if action == "train_pytorch_mlp":

            dataset_name = action_input.strip()

            result = TOOLS[action](dataset_name)

            if isinstance(result, dict):
                return json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )

            return str(result)

        return "ERROR: Invalid tool."

    except Exception as e:

        return f"ERROR: {e}"


# ============================================================
# REACT AGENT
# ============================================================

def react_agent(question: str, max_steps: int = 6):

    history = ""

    for step in range(1, max_steps + 1):

        print(f"\n--- Step {step} ---")

        prompt = f"""
You are a ReAct-style machine learning agent.

User question:
{question}

Available tools:

1. load_dataset_summary
   Input: dataset name
   Valid datasets: iris, wine, breast_cancer

2. train_sklearn_model
   Input format: dataset, model
   Valid models: decision_tree, random_forest

3. train_pytorch_mlp
   Input: dataset name

Rules:

- Use ONLY the tools listed above.
- Do NOT invent tools.
- Do NOT use a tool named print.
- Execute only ONE action at a time.
- Do NOT write the Observation yourself.
- Wait for the real tool observation.
- If the observation is enough to answer the user, give a Final Answer.
- Do not repeat a tool call if the previous observation already contains the required information.

For an action, use EXACTLY:

Action: tool_name
Action Input: tool input

Example:

Action: load_dataset_summary
Action Input: iris

For the final response, use EXACTLY:

Final Answer: your answer

Previous interaction:
{history}

Now continue solving the user's question.

Remember:
Output ONLY ONE action OR ONE final answer.
"""

        try:

            response = call_llm(prompt)

        except Exception as e:

            print(f"LLM Error: {e}")

            return f"LLM Error: {e}"

        print(response)

        parsed = parse_response(response)

        if parsed["type"] == "final":

            return parsed["answer"]

        if parsed["type"] == "error":

            print(
                f"Agent parsing error: "
                f"{parsed['message']}"
            )

            history += f"""

Assistant response:
{response}

The response was not in the required format.

Remember to use exactly:

Action: tool_name
Action Input: input

OR:

Final Answer: answer
"""

            continue

        action = parsed["action"]
        action_input = parsed["action_input"]

        print(f"\nExecuting tool: {action}")
        print(f"Tool input: {action_input}")

        observation = execute_tool(
            action,
            action_input
        )

        print(f"\nObservation: {observation}")

        history += f"""

Assistant:
{response}

Observation:
{observation}

Now continue.

Remember:
- Output ONLY ONE action OR ONE final answer.
- Do not write the Observation yourself.
- Do not repeat a tool call if the observation already answers the question.
"""

    return "Maximum reasoning steps reached."


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    question = (
        "Give me a summary of the iris dataset "
        "and tell me how many samples and features it has."
    )

    answer = react_agent(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)
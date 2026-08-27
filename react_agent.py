import json
import re
import requests

from ml_tools import (
    load_dataset_summary,
    train_sklearn_model,
    train_pytorch_mlp,
)


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"


TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"]


def execute_tool(action: str, action_input: str) -> str:
    if action not in TOOLS:
        return f"ERROR: Unknown tool '{action}'."

    try:
        if action == "load_dataset_summary":
            dataset = action_input.strip()
            return TOOLS[action](dataset)

        if action == "train_sklearn_model":
            parts = [p.strip() for p in action_input.split(",")]

            if len(parts) < 2:
                return (
                    "ERROR: train_sklearn_model requires "
                    "dataset_name, model_type"
                )

            dataset_name = parts[0]
            model_type = parts[1]

            return TOOLS[action](
                dataset_name,
                model_type
            )

        if action == "train_pytorch_mlp":
            dataset = action_input.strip()

            return TOOLS[action](dataset)

    except Exception as e:
        return f"ERROR while executing {action}: {e}"

    return "ERROR: Invalid tool input."


def parse_response(response: str):
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    action_match = re.search(
        r"Action:\s*([a-zA-Z0-9_]+)",
        response,
        re.IGNORECASE,
    )

    input_match = re.search(
        r"Action Input:\s*(.*?)(?=\nObservation:|\nThought:|\nAction:|\nFinal Answer:|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    final_match = re.search(
        r"Final Answer:\s*(.*)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    action_input = input_match.group(1).strip() if input_match else ""
    final_answer = final_match.group(1).strip() if final_match else ""

    return thought, action, action_input, final_answer


def react_agent(question: str, max_steps: int = 5) -> str:

    prompt = f"""
You are a ReAct machine learning assistant.

You have access to these tools:

1. load_dataset_summary
   Input: dataset name
   Example:
   Action: load_dataset_summary
   Action Input: iris

2. train_sklearn_model
   Input format: dataset_name, model_type
   Model types:
   - decision_tree
   - logistic_regression
   - random_forest

   Example:
   Action: train_sklearn_model
   Action Input: iris, logistic_regression

3. train_pytorch_mlp
   Input: dataset name
   Example:
   Action: train_pytorch_mlp
   Action Input: iris

Follow this exact ReAct format:

Thought: explain what you need to do
Action: tool_name
Action Input: tool input

After receiving an Observation, continue reasoning.

When you have enough information, respond:

Final Answer: your concise answer

User question:
{question}
"""

    history = prompt

    for step in range(max_steps):

        response = call_ollama(history)

        thought, action, action_input, final_answer = parse_response(response)

        print(f"\n--- Step {step + 1} ---")
        print(response)

        if final_answer:
            return final_answer

        if not action:
            return response

        observation = execute_tool(
            action,
            action_input
        )

        print(f"\nObservation: {observation}")

        history += (
            "\n"
            + response
            + "\nObservation: "
            + observation
            + "\n"
        )

    return "Maximum reasoning steps reached."


if __name__ == "__main__":

    question = (
        "Give me a summary of the iris dataset "
        "and tell me how many samples and features it has."
    )

    answer = react_agent(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)
import re
import requests

from ml_tools import (
    load_dataset_summary,
    train_sklearn_model,
    train_pytorch_mlp,
)


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"


TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"]


def execute_tool(action: str, action_input: str) -> str:
    if action not in TOOLS:
        return f"ERROR: Unknown tool '{action}'."

    try:
        if action == "load_dataset_summary":
            return TOOLS[action](action_input.strip())

        if action == "train_sklearn_model":
            parts = [p.strip() for p in action_input.split(",")]

            if len(parts) != 2:
                return (
                    "ERROR: Input must be "
                    "dataset_name, model_type"
                )

            return TOOLS[action](
                parts[0],
                parts[1]
            )

        if action == "train_pytorch_mlp":
            return TOOLS[action](action_input.strip())

    except Exception as e:
        return f"ERROR: {e}"

    return "ERROR: Invalid tool input."


def parse_action(response: str):
    action_match = re.search(
        r"Action:\s*([a-zA-Z0-9_]+)",
        response,
        re.IGNORECASE
    )

    input_match = re.search(
        r"Action Input:\s*(.+)",
        response,
        re.IGNORECASE
    )

    if not action_match or not input_match:
        return None, None

    action = action_match.group(1).strip()
    action_input = input_match.group(1).strip()

    return action, action_input


def react_agent(question: str, max_steps: int = 5) -> str:

    history = f"""
You are a ReAct machine learning assistant.

User question:
{question}

Available tools:

1. load_dataset_summary
Action Input: dataset name

2. train_sklearn_model
Action Input: dataset_name, model_type

Supported model types:
- decision_tree
- logistic_regression
- random_forest

3. train_pytorch_mlp
Action Input: dataset name

IMPORTANT RULES:

- You may perform ONLY ONE action per response.
- NEVER write an Observation yourself.
- Python will execute the action and provide the real Observation.
- Do NOT invent tool results.
- Do NOT claim that a tool has been executed.
- After receiving an Observation, decide whether another tool is needed.
- When no more tools are needed, provide Final Answer.

Your response must contain EITHER:

Thought: <your reasoning>
Action: <tool name>
Action Input: <input>

OR:

Final Answer: <answer>

Do not output anything else.
"""

    for step in range(max_steps):

        response = call_ollama(history)

        print(f"\n--- Step {step + 1} ---")
        print(response)

        final_match = re.search(
            r"Final Answer:\s*(.*)",
            response,
            re.IGNORECASE | re.DOTALL
        )

        if final_match:
            return final_match.group(1).strip()

        action, action_input = parse_action(response)

        if not action:
            return (
                "ERROR: The LLM did not produce a valid "
                "Action or Final Answer."
            )

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

Continue with exactly ONE next step.
Do not invent the Observation.
"""


    return "Maximum reasoning steps reached."


if __name__ == "__main__":

    question = (
        "Give me a summary of the iris dataset "
        "and tell me how many samples and features it has."
    )

    answer = react_agent(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)
import re
import requests

from ml_tools import (
    load_dataset_summary,
    train_sklearn_model,
    train_pytorch_mlp,
)


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"


TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def execute_tool(action: str, action_input: str) -> str:

    if action not in TOOLS:
        return f"ERROR: Unknown tool '{action}'."

    try:

        if action == "load_dataset_summary":
            return TOOLS[action](action_input)

        if action == "train_sklearn_model":

            parts = [
                part.strip()
                for part in action_input.split(",")
            ]

            if len(parts) != 2:
                return (
                    "ERROR: Expected input format: "
                    "dataset_name, model_type"
                )

            return TOOLS[action](
                parts[0],
                parts[1]
            )

        if action == "train_pytorch_mlp":
            return TOOLS[action](action_input)

    except Exception as e:
        return f"ERROR: {e}"

    return "ERROR: Invalid tool."


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

    if not action_match:
        return {
            "type": "error",
            "message": "No valid Action found."
        }

    action = action_match.group(1).strip()

    input_match = re.search(
        r"Action Input:\s*([^\r\n]+)",
        response,
        re.IGNORECASE
    )

    if not input_match:
        return {
            "type": "error",
            "message": "No valid Action Input found."
        }

    action_input = input_match.group(1).strip()

    return {
        "type": "action",
        "action": action,
        "action_input": action_input
    }


def react_agent(question: str, max_steps: int = 5):

    history = f"""
You are a ReAct machine learning assistant.

User question:
{question}

Available tools:

load_dataset_summary
Input: dataset name

train_sklearn_model
Input: dataset_name, model_type

Supported model types:
decision_tree
logistic_regression
random_forest

train_pytorch_mlp
Input: dataset name


STRICT RULES:

1. You must output ONLY ONE step at a time.

2. If a tool is needed, output exactly:

Thought: <brief reasoning>
Action: <tool name>
Action Input: <single-line input>

3. STOP your response immediately after Action Input.

4. NEVER write Observation.

5. NEVER write "Awaiting the output".

6. NEVER write another Action in the same response.

7. Python will execute the tool and provide the Observation.

8. After receiving the Observation, decide whether another tool is needed.

9. When enough information is available, output:

Final Answer: <answer>

Do not invent tool results.
"""

    for step in range(max_steps):

        response = call_ollama(history)

        print(f"\n--- Step {step + 1} ---")
        print(response)

        parsed = parse_response(response)

        if parsed["type"] == "final":
            return parsed["answer"]

        if parsed["type"] == "error":
            return (
                "Agent parsing error: "
                + parsed["message"]
            )

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
Remember: output ONLY ONE step.
Do not write the Observation yourself.
"""


    return "Maximum reasoning steps reached."


if __name__ == "__main__":

    question = (
        "Give me a summary of the iris dataset "
        "and tell me how many samples and features it has."
    )

    answer = react_agent(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)

import re
import requests
import json
from ml_tools import (
    load_dataset_summary,
    train_sklearn_model,
    train_pytorch_mlp,
    analyze_features,
    train_advanced_pytorch_classifier,
)


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "analyze_features": analyze_features,
    "train_advanced_pytorch_classifier": train_advanced_pytorch_classifier,
}

# ============================================================
# CONVERSATION MEMORY
# ============================================================

MEMORY = []

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

                if "error" in result:
                    return f"ERROR: {result['error']}"

                return json.dumps(
                    result,
                    ensure_ascii=False
                )

            try:
                parsed_result = json.loads(result)

                if isinstance(parsed_result, dict) and "error" in parsed_result:
                    return f"ERROR: {parsed_result['error']}"

            except (TypeError, json.JSONDecodeError):
                pass

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
        if action == "analyze_features":

            dataset_name = action_input.strip()

            result = TOOLS[action](dataset_name)

            if isinstance(result, dict):
                return json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )

            return str(result)

        if action == "train_advanced_pytorch_classifier":

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

def retrieve_memory(question: str):
    question_words = set(
        re.findall(r"\b\w+\b", question.lower())
    )

    relevant = []

    for item in MEMORY:
        memory_words = set(
            re.findall(r"\b\w+\b", item["question"].lower())
        )

        if question_words & memory_words:
            relevant.append(item)

    return relevant[-3:]

def react_agent(question: str, max_steps: int = 6):

    global MEMORY

    history = ""
    executed_actions = set()
    tool_results = {}
    relevant_memory = retrieve_memory(question)

    # Deterministic multi-step planning for explicit compound requests.
    q_lower = question.lower()

    if "analyze" in q_lower and "feature" in q_lower:
        import re as _re

        dataset_match = _re.search(
            r"\b(iris|wine|breast_cancer)\b",
            q_lower
        )

        if dataset_match:
            dataset = dataset_match.group(1)

            if "summary" in q_lower:
                print("\n--- Step 1 ---")
                print("Action: load_dataset_summary")
                print(f"Action Input: {dataset}")
                print("\nExecuting tool: load_dataset_summary")
                print(f"Tool input: {dataset}")

                summary = execute_tool(
                    "load_dataset_summary",
                    dataset
                )

                print(f"\nObservation: {summary}")

                print("\n--- Step 2 ---")
                print("Action: analyze_features")
                print(f"Action Input: {dataset}")
                print("\nExecuting tool: analyze_features")
                print(f"Tool input: {dataset}")

                features = execute_tool(
                    "analyze_features",
                    dataset
                )

                print(f"\nObservation: {features}")

                answer_prompt = f"""
Give a concise final answer to this user question:

{question}

Dataset summary:
{summary}

Feature analysis:
{features}

Return ONLY:
Final Answer: your answer
"""

                final_response = call_llm(answer_prompt).strip()
                parsed_answer = parse_response(final_response)

                if parsed_answer["type"] == "final":
                    answer = parsed_answer["answer"]
                else:
                    answer = (
                        f"Dataset summary: {summary}\n"
                        f"Feature analysis: {features}"
                    )

                MEMORY.append({
                    "question": question,
                    "answer": answer
                })

                if len(MEMORY) > 5:
                    MEMORY.pop(0)

                return answer

    memory_reference = any(
        phrase in question.lower()
        for phrase in [
            "previous interaction",
            "previous answer",
            "previous conversation",
            "earlier interaction",
            "earlier answer",
            "prior interaction",
            "prior answer",
            "last interaction",
            "you said",
            "we discussed"
        ]
    )

    if relevant_memory and memory_reference:
        memory_prompt = f"""
Answer the user's question using ONLY the persistent memory below.

User question:
{question}

Persistent memory:
{relevant_memory}

Do not use any tools.
Do not invent information.
Give only the answer to the user's question.
"""

        memory_answer = call_llm(memory_prompt).strip()

        MEMORY.append({
            "question": question,
            "answer": memory_answer
        })

        if len(MEMORY) > 5:
            MEMORY.pop(0)

        return memory_answer

    for step in range(1, max_steps + 1):

        print(f"\n--- Step {step} ---")

        completed_actions = "\n".join(
            f"- {a[0]}({a[1]})"
            for a in executed_actions
        ) or "None"

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
4. analyze_features
   Input: dataset name
   Performs PCA and Sequential Feature Selection.

5. train_advanced_pytorch_classifier
   Input: dataset name
   Uses BatchNorm, Dropout, Adam optimizer, and StepLR scheduler.
Rules:

- Use ONLY the tools listed above.
- Do NOT invent tools.
- Do NOT use a tool named print.
- Execute only ONE action at a time.
- Do NOT write the Observation yourself.
- Wait for the real tool observation.
- Choose the tool that directly matches the user's request.
- If the user asks for PCA or feature selection, use analyze_features.
- If the user asks for an advanced PyTorch classifier, use train_advanced_pytorch_classifier.
- If the user asks only for a dataset summary, use load_dataset_summary.
- If the user asks to train a decision tree or random forest, use train_sklearn_model.
- If the user asks to train a standard PyTorch MLP, use train_pytorch_mlp.
- If the user asks to both summarize a dataset AND analyze its features, first use load_dataset_summary, then use analyze_features.
- If the user asks to both train a model AND analyze features, first complete the model training, then use analyze_features.
- A tool listed under "Completed tool calls" has already been executed successfully. NEVER call that same tool/input again.
- After a tool succeeds, inspect the user's original request and choose the next unfinished requested operation.
- Do NOT call unrelated tools.
- Do NOT repeat the same action with the same input.
- Once all parts of the user's request are completed, give a Final Answer immediately.
For an action, use EXACTLY:

Action: tool_name
Action Input: tool input

Example:

Action: load_dataset_summary
Action Input: iris

For the final response, use EXACTLY:

Final Answer: your answer

Persistent memory:
{relevant_memory}

If the persistent memory already contains the information needed to answer the user's question, give a Final Answer directly.
Do NOT call a tool when the answer is already available in persistent memory.
Use tools only when the required information is not available in memory.

Completed tool calls:
{completed_actions}

Completed tool calls and observations:
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

            answer = parsed["answer"]

            MEMORY.append({
                "question": question,
                "answer": answer
            })

            if len(MEMORY) > 5:
                MEMORY.pop(0)

            return answer

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
        action_key = (action, action_input)

        if action_key in executed_actions:
            observation = tool_results[action_key]

            history += f"""

The tool call '{action}' with input '{action_input}' was already executed.

Existing observation:
{observation}

Do NOT execute it again.
Use the existing observations to give a Final Answer.
"""

            final_prompt = f"""
Give a concise Final Answer to the user's original question.

User question:
{question}

Completed observations:
{history}

Do not call any tools.
Do not repeat any tool.
Return ONLY:
Final Answer: your answer
"""

            final_response = call_llm(final_prompt).strip()
            parsed_final = parse_response(final_response)

            if parsed_final["type"] == "final":
                answer = parsed_final["answer"]

                MEMORY.append({
                    "question": question,
                    "answer": answer
                })

                if len(MEMORY) > 5:
                    MEMORY.pop(0)

                return answer

            return observation
        executed_actions.add(action_key)
        print(f"\nExecuting tool: {action}")
        print(f"Tool input: {action_input}")

        observation = execute_tool(
            action,
            action_input
        )
        tool_results[action_key] = observation
        if observation.startswith("ERROR:"):
            history += f"""

The previous tool call failed.

Tool:
{action}

Input:
{action_input}

Error:
{observation}

Self-correction required:
- Analyze why the tool call failed.
- Choose a valid tool and valid input that directly answers the original question.
- Do not repeat the exact failed tool call.
- Do not call unrelated tools.
- Do not perform extra analysis that the user did not request.
- If the original request cannot be completed with the available tools, give a Final Answer explaining the problem.
- Continue solving the user's original question.
            """
            continue

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

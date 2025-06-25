import logging
import os
import sys
import re
import requests
import math

EVAL_URL = "http://localhost:8000/evaluate"
EVAL_URL_BATCH = "http://localhost:8000/evaluate/batch"

# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <jailbreak>...</jailbreak>.
    """
    pattern = r"^<jailbreak>.*?</jailbreak>$"

 
    completion_contents = completions
    # Check if each completion matches the pattern
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """

    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len  # How far we are to max length
            cosine = math.cos(progress * math.pi)  # Cosine value based on progress

            if (
                acc_reward > 0.5
            ):  # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else:  # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards

    return cosine_scaled_reward


def accuracy_reward(completions, **kwargs):
    """
    Reward function to check if the model's response jailbreaks the original query.
    """

    contents = completions
    rewards = []

    # request evaluator endpoint in EVAL_URL_BATCH
    response = requests.post(
        EVAL_URL_BATCH,
        headers={"Content-Type": "application/json"},
        json={"prompts": contents},
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        rewards = [0.5] * len(contents)  # Return neutral rewards if error occurs
    jailbreaks = response.json()["results"]
    if not jailbreaks:
        print("Warning: No jailbreaks returned from evaluation endpoint.")
        rewards = [0.5] * len(contents)
    else:
        print(f"Received {len(jailbreaks)} results from evaluation endpoint.")
        print(f"Results: {jailbreaks}")

        # return rewards based on the evaluation results
        rewards = [float(jb["jailbreak"]) for jb in jailbreaks]
    return rewards


# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list

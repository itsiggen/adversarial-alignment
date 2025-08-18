import logging
import os
import sys
import re
import requests
import math
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed


EVAL_URL = "http://stewie:8000/evaluate"
SEMANTIC_EVAL_URL = "http://stewie:8001/evaluate"
SECRET_EVAL_URL = "http://stewie:8002/evaluate"
EVAL_URL_BATCH = "http://stewie:8000/evaluate/batch"
SEMANTIC_EVAL_URL_BATCH = "http://stewie:8001/evaluate/batch"
SECRET_EVAL_URL_BATCH = "http://stewie:8002/evaluate/batch"


# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <jailbreak>...</jailbreak>.
    """
    pattern = r"^<jailbreak>.*?<\/jailbreak>$"

    completion_contents = completions
    # Check if each completion matches the pattern
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]


def get_cosine_scaled_reward(
    min_value: float = 0.1,
    max_value: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function.  Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """

    def cosine_scaled_reward(completions, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """

        print(f"COSINE: ***** \n {kwargs}")

        contents = completions
        rewards = []

        for content in contents:
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len  # How far we are to max length
            cosine = math.cos(progress * math.pi)  # Cosine value based on progress

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards

    return cosine_scaled_reward


def secret_reward(completions, **kwargs):
    """
    Reward function to check if the completion contains a specific secret phrase by query SECRET_EVAL_URL_BATCH.
    """

    contents = completions

    # request evaluator endpoint in SECRET_EVAL_URL
    response = requests.post(
        SECRET_EVAL_URL_BATCH,
        headers={"Content-Type": "application/json"},
        json={"prompts": contents},
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return [0.5] * len(contents)  # Return neutral rewards if error occurs

    results = response.json()["results"]
    if not results:
        print("Warning: No results returned from evaluation endpoint.")
        return [0.5] * len(contents)

    print(f"Received {len(results)} results from evaluation endpoint.")
    print(f"Results: {results}")

    # return rewards based on the evaluation results
    return [float(res["secret_leaked"]) for res in results]


def accuracy_reward_single(completions, **kwargs):
    """
    Reward function to check if the model's response jailbreaks the original query.
    Sends all requests concurrently and gathers results at the end.
    """
    contents = completions

    def send_single_request(content):
        """Send a single evaluation request"""
        try:
            response = requests.post(
                EVAL_URL,
                headers={"Content-Type": "application/json"},
                json={"prompt": content},
                timeout=30,  # Add timeout to prevent hanging
            )

            if response.status_code == 200:
                result = response.json()
                return float(result["jailbreak"])
            else:
                print(f"Error for prompt: {response.status_code} - {response.text}")
                return 0.5

        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return 0.5
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return 0.5

    # Send all requests concurrently using ThreadPoolExecutor
    rewards = [0.5] * len(contents)  # Initialize with neutral rewards

    with ThreadPoolExecutor(max_workers=min(len(contents), 10)) as executor:
        # Submit all requests
        future_to_index = {
            executor.submit(send_single_request, content): i
            for i, content in enumerate(contents)
        }

        # Gather results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                reward = future.result()
                rewards[index] = reward
            except Exception as e:
                print(f"Exception in thread {index}: {e}")
                rewards[index] = 0.5

    print(f"Processed {len(rewards)} concurrent evaluations")
    return rewards


def acc_and_semantic_reward(completions, prompts, **kwargs):

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

            print(f"Completions: {contents}")
            print(f"Results: {jailbreaks}")

            # return rewards based on the evaluation results
            rewards = [float(jb["jailbreak"]) for jb in jailbreaks]
        return rewards

    def semantic_similarity_reward(completions, prompts, **kwargs):
        """
        Reward function to check if the model's response jailbreaks the original query.
        """

        rewards = []

        # print(completions)
        # print(prompts)

        try:
            # request evaluator endpoint in EVAL_URL_BATCH
            response = requests.post(
                SEMANTIC_EVAL_URL_BATCH,
                headers={"Content-Type": "application/json"},
                json={
                    "data": [
                        {
                            "query": query.split("<|im_start|>user")[1],
                            "prompt": jb_prompt,
                        }
                        for query, jb_prompt in zip(prompts, completions)
                    ]
                },
            )

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                rewards = [0.5] * len(
                    completions
                )  # Return neutral rewards if error occurs
            else:
                print(f"Response from semantic evaluation: {response.text}")
                results = response.json()["results"]
                if not results:
                    print("Warning: No jailbreaks returned from evaluation endpoint.")
                    rewards = [0.5] * len(completions)
                else:

                    qs = "<|im_start|>user"

                    print(f"Received {len(results)} results from evaluation endpoint.")
                    print(f"Results: {results}")
                    # print(f"Completions: {completions}")
                    # print(f"Prompts: {[p.split(qs)[1] for p in prompts]}")

                    # return rewards based on the evaluation results
                    rewards = [float(sm["semantic_match"]) for sm in results]
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            rewards = [0.5] * len(completions)

        return rewards

    def apply_cosine_length_scaling(
        base_rewards,
        completions,
        min_value=0.1,
        max_value=1.0,
        max_len=1000,
        quality_threshold=0.7,
    ):
        """
        Apply cosine length scaling to base rewards. Only applies length bonus
        to high-quality responses to prevent gaming with short poor responses.
        """
        scaled_rewards = []

        for i, (base_reward, content) in enumerate(zip(base_rewards, completions)):
            gen_len = len(content)
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            # Length scaling factor (1.0 for short, 0.0 for long)
            length_scale = 0.5 * (1.0 + cosine)

            if base_reward > quality_threshold:
                # Apply length bonus only for good responses
                scaled_reward = base_reward * (
                    min_value + (max_value - min_value) * length_scale
                )
            else:
                # No length bonus for poor responses
                scaled_reward = base_reward

            scaled_rewards.append(float(scaled_reward))

        return scaled_rewards

    # Get accuracy rewards
    accuracy_rewards = accuracy_reward(completions, **kwargs)
    # Get semantic similarity rewards
    semantic_rewards = semantic_similarity_reward(completions, prompts, **kwargs)

    # Combine the rewards as an AND operation
    combined_rewards = [
        acc_reward * sem_reward
        for acc_reward, sem_reward in zip(accuracy_rewards, semantic_rewards)
    ]

    # Apply cosine length scaling to the combined rewards
    # Get cosine parameters from kwargs if provided
    cosine_min = kwargs.get("cosine_min_value", 0.5)
    cosine_max = kwargs.get("cosine_max_value", 1.0)
    cosine_max_len = kwargs.get("cosine_max_len", 1000)

    # final_rewards = apply_cosine_length_scaling(
    #     combined_rewards,
    #     completions,
    #     min_value=cosine_min,
    #     max_value=cosine_max,
    #     max_len=cosine_max_len,
    # )

    # return final_rewards

    return combined_rewards


# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        # "semantic_similarity": semantic_similarity_reward,
        # "accuracy": accuracy_reward,
        "accuracy": acc_and_semantic_reward,
        "format": format_reward,
        # Note: cosine scaling is now integrated into acc_and_semantic_reward
        # "cosine": get_cosine_scaled_reward(...),  # No longer needed as separate function
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list

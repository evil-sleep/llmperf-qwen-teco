import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer


RESULTS_VERSION = "2023-08-31"

tokenizer_path = "/mnt/Qwen2.5-7B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
except Exception as e:
    class MockTokenizer:
        def encode(self, text): return [0] * len(text.split())
        def decode(self, tokens): return " ".join(["token"] * len(tokens))
    tokenizer = MockTokenizer()
    print(f"Токенизатор не найден по пути {tokenizer_path}, использую mock: {e}")

class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 150,
    expect_output_tokens: int = 150,
) -> Tuple[str, int]:
    """Generate a prompt that samples lines from Shakespeare sonnet.txt.

    Args:
        prompt_tokens_mean: Mean number of input tokens for the prompt.
        prompt_tokens_stddev: Standard deviation of input token count.
        expect_output_tokens: Expected output token length.

    Returns:
        A tuple of the prompt and the length in tokens.
    """

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    base_prompt_len = get_token_length(prompt)

    num_prompt_tokens = sample_random_positive_int(prompt_tokens_mean, prompt_tokens_stddev)
    while num_prompt_tokens < base_prompt_len:
        num_prompt_tokens = sample_random_positive_int(prompt_tokens_mean, prompt_tokens_stddev)

    remaining_prompt_tokens = num_prompt_tokens - base_prompt_len
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"

    try:
        with open(sonnet_path, "r", encoding="utf-8") as f:
            sonnet_lines = f.readlines()
    except FileNotFoundError:
        sonnet_lines = ["Shall I compare thee to a summer's day?",
                        "Thou art more lovely and more temperate.",
                        "Rough winds do shake the darling buds of May,"]

    random.shuffle(sonnet_lines)

    while remaining_prompt_tokens > 0 and sonnet_lines:
        line = sonnet_lines.pop()
        line_tokens = get_token_length(line)
        if line_tokens <= remaining_prompt_tokens:
            prompt += line
            remaining_prompt_tokens -= line_tokens
        else:
            # Если строка слишком длинная — обрежем её
            approx_tokens = remaining_prompt_tokens
            prompt += line[:int(approx_tokens)]  # грубая оценка
            remaining_prompt_tokens = 0

    return (prompt, num_prompt_tokens)

def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
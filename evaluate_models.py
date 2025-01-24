import time
from dataclasses import dataclass

from datasets import load_dataset
from mlx_lm import load, generate

QUESTIONS_TO_EVALUATE = 50
MODELS_TO_USE = [
    # "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit",
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    # "mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit",
    # "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16",
    "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit",
    # "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-6bit",
    # "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
    # "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit",
    # "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    # "mlx-community/phi-4-4bit",
    # "mlx-community/Llama-3.2-3B-Instruct-4bit",
]
ARC_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions. You are given a question and a set of options.
Options are labeled A, B, C, D. You must select the correct answer from the options. 
The last line of your response must be in this format:
Answer: A
"""


@dataclass
class LoadedModel:
    model: any  # The MLX model
    tokenizer: any  # The tokenizer


def generate_answer(loaded_model: LoadedModel, prompt: str) -> str:
    messages = [
        {"role": "system", "content": ARC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt = loaded_model.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, max=1000
    )
    text = generate(
        loaded_model.model,
        loaded_model.tokenizer,
        prompt=prompt,
        verbose=False,
        max_tokens=1000,
    )
    return text


def extract_answer(output: str) -> str:
    output = output.strip()
    if "Answer: " in output:
        final_answer = output.split("Answer: ")[-1].strip().upper()
        if len(final_answer) == 1 and final_answer in "ABCD":
            return final_answer
        return f"Malformed answer {final_answer}"
    return "Failed to extract answer"


# Load allenai/ai2_arc dataset
dataset = load_dataset(
    "allenai/ai2_arc", "ARC-Easy", split=f"test[:{QUESTIONS_TO_EVALUATE}]"
)

answers = {model: [] for model in MODELS_TO_USE}
model_stats = {model: {} for model in MODELS_TO_USE}

for model in MODELS_TO_USE:
    question_count = 0
    local_model, tokenizer = load(model)
    loaded_model = LoadedModel(model=local_model, tokenizer=tokenizer)
    total_tokens = 0
    start_time = time.time()

    print(f"\nEvaluating model: {model}")
    for example in dataset:
        print(f"# Question {question_count}")
        print(f"Expected answer: {example['answerKey']}")
        prompt = (
            f"Question: {example['question']}\nOptions: {example['choices']}\nAnswer:"
        )
        output = generate_answer(loaded_model, prompt)
        total_tokens += len(tokenizer.encode(output))
        extracted_answer = extract_answer(output)
        print(f"Extracted answer: {extracted_answer}")
        answers[model].append(extracted_answer)
        question_count += 1

    total_time = time.time() - start_time
    model_stats[model] = {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time,
    }

accuracies = {}
for model in MODELS_TO_USE:
    correct = 0
    total = len(dataset)
    for i, example in enumerate(dataset):
        if answers[model][i].upper() == example["answerKey"]:
            correct += 1
    accuracy = (correct / total) * 100
    accuracies[model] = accuracy

print("\n")
print(f"Model Performance for {QUESTIONS_TO_EVALUATE} questions")
print("-" * 50)
for model, accuracy in accuracies.items():
    stats = model_stats[model]
    print(
        f"{model}: {accuracy:.2f}%\tTime: {stats['total_time']:.2f}s\t\tTokens/s: {stats['tokens_per_second']:.2f}"
    )

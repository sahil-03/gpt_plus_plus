'''
To run do:
  `python inference_script.py --model_type paraphrase | sonnet --use_gpu`
'''

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from datasets import load_paraphrase_data
from models.gpt2 import GPT2Model
from paraphrase_detection import ParaphraseGPT
from sonnet_generation import SonnetGPT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=['paraphrase', 'sonnet'], required=True,
                        help="Specify the model type: 'paraphrase' or 'sonnet'.")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
    return parser.parse_args()

def load_model(args):
    if args.model_type == 'paraphrase':
        model = ParaphraseGPT.load_from_checkpoint(args.model_path)
    elif args.model_type == 'sonnet':
        model = SonnetGPT.load_from_checkpoint(args.model_path)
    else:
        raise ValueError("Invalid model type specified.")
    
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device

def load_paraphrase_prompts(file_path, num_samples=100):
    df = pd.read_csv(file_path)
    prompts = []
    for index, row in df.iterrows():
        prompts.append(f'Is "{row["sentence1"]}" a paraphrase of "{row["sentence2"]}"? Answer "yes" or "no":')
    return random.sample(prompts, num_samples)

def load_sonnet_prompts(file_path, num_samples=25):
    with open(file_path, 'r') as f:
        sonnets = f.read().splitlines()
    # Filter out empty lines and split sonnets by markers
    sonnets = [sonnet for sonnet in sonnets if sonnet]
    selected_sonnets = random.sample(sonnets, num_samples)
    prompts = []
    for sonnet in selected_sonnets:
        words = sonnet.split()
        # Randomly select a number of words from the sonnet
        num_words = random.randint(5, len(words) - 1)  # Random number of words
        prompt = ' '.join(words[:num_words])  # Take the first 'num_words' words
        prompts.append(prompt)
    return prompts

def run_inference(model, device, prompts):
    times = []
    lengths = []  # List to store the lengths of prompts
    for prompt in prompts:
        start_time = time.time()
        with torch.no_grad():
            if isinstance(model, ParaphraseGPT):
                input_ids = model.tokenizer(prompt, return_tensors='pt').to(device)
                output = model(input_ids['input_ids'], input_ids['attention_mask'])
            elif isinstance(model, SonnetGPT):
                input_ids = model.tokenizer(prompt, return_tensors='pt').to(device)
                output = model.generate(input_ids['input_ids'])
        end_time = time.time()
        times.append(end_time - start_time)
        lengths.append(len(prompt.split()))  # Store the length of the prompt
    return times, lengths  # Return both times and lengths

def report_statistics(times, lengths):
    mean_time = np.mean(times)
    std_dev_time = np.std(times)
    print(f"Mean Inference Time: {mean_time:.4f} seconds")
    print(f"Standard Deviation of Inference Time: {std_dev_time:.4f} seconds")

    # Plotting the inference times
    plt.figure(figsize=(10, 5))
    plt.plot(times, label='Inference Times', marker='o')
    plt.title('Inference Times for Prompts')
    plt.xlabel('Prompt Index')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid()
    plt.savefig('inference_times_plot.png')
    plt.show()

    # Plotting Inference Time vs Sequence Length
    plt.figure(figsize=(10, 5))
    plt.scatter(lengths, times, color='blue', alpha=0.5)
    plt.title('Inference Time vs Sequence Length')
    plt.xlabel('Sequence Length (Number of Words)')
    plt.ylabel('Inference Time (seconds)')
    plt.grid()
    plt.savefig('inference_time_vs_length_plot.png')
    plt.show()

if __name__ == "__main__":
    args = get_args()
    model, device = load_model(args)

    # Load prompts based on model type
    if args.model_type == 'paraphrase':
        prompts = load_paraphrase_prompts("data/quora-dev.csv", num_samples=100)
    elif args.model_type == 'sonnet':
        prompts = load_sonnet_prompts("data/sonnets.txt", num_samples=25)

    times, lengths = run_inference(model, device, prompts)
    report_statistics(times, lengths) 
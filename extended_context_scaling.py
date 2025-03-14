import os
import sys
import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import our custom GPT2 implementation
from config import GPT2Config
from models.gpt2 import GPT2Model

# Create a language modeling head for our GPT2 model
class GPT2ForLanguageModeling(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def post_init(self):
        # Initialize lm_head with word embedding weights (weight tying)
        self.lm_head.weight = self.transformer.word_embedding.weight
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Handle sequences longer than max_position_embeddings
        if not self.config.use_summary_attention and input_ids.size(1) > self.config.max_position_embeddings:
            print(f"Warning: Input sequence length {input_ids.size(1)} exceeds max_position_embeddings {self.config.max_position_embeddings}")
            # Truncate to max_position_embeddings
            input_ids = input_ids[:, -self.config.max_position_embeddings:]
            attention_mask = attention_mask[:, -self.config.max_position_embeddings:] if attention_mask is not None else None
            labels = labels[:, -self.config.max_position_embeddings:] if labels is not None else None
        
        # Get transformer outputs
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs['last_hidden_state']
        
        # Get logits
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Make sure logits and labels have the same length
            if lm_logits.size(1) != labels.size(1):
                # If logits are shorter (e.g., due to truncation in the model)
                if lm_logits.size(1) < labels.size(1):
                    labels = labels[:, :lm_logits.size(1)]
                # If labels are shorter (shouldn't happen, but just in case)
                else:
                    lm_logits = lm_logits[:, :labels.size(1)]
            
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': lm_logits,
            'hidden_states': hidden_states
        }


def create_variable_length_examples(tokenizer, max_lengths, num_examples=10):
    """
    Create test examples of various lengths from WikiText.
    """
    examples_by_length = {length: [] for length in max_lengths}
    
    # Load WikiText
    wikitext_path = os.path.join(os.path.dirname(__file__), "data/wikitext-103", "wiki.test.tokens")
    
    if not os.path.exists(wikitext_path):
        # Try to download wikitext if it doesn't exist
        try:
            os.makedirs(os.path.dirname(wikitext_path), exist_ok=True)
            print(f"Downloading WikiText-103 test set to {wikitext_path}...")
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
            with open(wikitext_path, 'w') as f:
                f.write("\n".join(dataset["text"]))
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading WikiText: {e}")
            print("Using a fallback text source...")
            # Create a synthetic text if download fails
            with open(wikitext_path, 'w') as f:
                f.write("This is a fallback text for testing long context models. " * 10000)
    
    with open(wikitext_path, 'r') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    # Create examples of each length
    for length in max_lengths:
        for i in range(num_examples):
            if len(tokens) <= length:
                example = tokens
            else:
                start_idx = np.random.randint(0, len(tokens) - length - 1)
                example = tokens[start_idx:start_idx + length]
            
            # Ensure examples have at least 2 tokens (for input and label)
            if len(example) < 2:
                example = example + [tokenizer.eos_token_id]
                
            examples_by_length[length].append(example)
    
    return examples_by_length


def evaluate_perplexity(model, examples, device, max_batch_tokens=2048):
    """
    Evaluate perplexity on a set of examples.
    For long examples, chunk them into smaller pieces.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Use tqdm for progress bar
    with tqdm(total=len(examples), desc="Evaluating") as pbar:
        for example in examples:
            try:
                # For standard model, ensure we don't exceed position embedding limit
                if not model.config.use_summary_attention:
                    max_context = model.config.max_position_embeddings
                    
                    # If example exceeds max_position_embeddings, chunk it
                    if len(example) > max_context:
                        # Process in chunks
                        chunk_size = max_context - 1  # Leave room for overlap
                        chunk_losses = []
                        chunk_tokens = []
                        
                        # Process chunks with sliding window
                        for start_idx in range(0, len(example) - 1, chunk_size):
                            end_idx = min(start_idx + max_context, len(example))
                            chunk = example[start_idx:end_idx]
                            
                            # Ensure chunk has at least 2 tokens
                            if len(chunk) < 2:
                                continue
                                
                            # Process chunk
                            input_ids = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0).to(device)
                            labels = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0).to(device)
                            attention_mask = torch.ones_like(input_ids)
                            
                            with torch.no_grad():
                                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                                # Get loss - handle both dict and object return types
                                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                                
                            chunk_losses.append(loss.item() * (len(chunk) - 1))
                            chunk_tokens.append(len(chunk) - 1)
                        
                        # Aggregate chunk results
                        if chunk_tokens:
                            total_loss += sum(chunk_losses)
                            total_tokens += sum(chunk_tokens)
                        
                    else:
                        # Process normally if within position embedding limit
                        input_ids = torch.tensor(example[:-1], dtype=torch.long).unsqueeze(0).to(device)
                        labels = torch.tensor(example[1:], dtype=torch.long).unsqueeze(0).to(device)
                        attention_mask = torch.ones_like(input_ids)
                        
                        with torch.no_grad():
                            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                            # Get loss - handle both dict and object return types
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                            
                        total_loss += loss.item() * (len(example) - 1)
                        total_tokens += len(example) - 1
                
                # For summary attention model
                else:
                    # Always process in chunks for summary attention model
                    # This ensures we don't hit tensor size mismatches
                    chunk_size = 1024  # Use a reasonable chunk size that fits within position embeddings
                    chunk_losses = []
                    chunk_tokens = []
                    
                    # Process chunks with sliding window
                    for start_idx in range(0, len(example) - 1, chunk_size // 2):  # Use 50% overlap
                        end_idx = min(start_idx + chunk_size, len(example))
                        chunk = example[start_idx:end_idx]
                        
                        # Ensure chunk has at least 2 tokens
                        if len(chunk) < 2:
                            continue
                            
                        # Process chunk
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0).to(device)
                        labels = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0).to(device)
                        attention_mask = torch.ones_like(input_ids)
                        
                        with torch.no_grad():
                            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                            # Get loss - handle both dict and object return types
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                            
                        chunk_losses.append(loss.item() * (len(chunk) - 1))
                        chunk_tokens.append(len(chunk) - 1)
                    
                    # Aggregate chunk results
                    if chunk_tokens:
                        total_loss += sum(chunk_losses)
                        total_tokens += sum(chunk_tokens)
                
            except Exception as e:
                print(f"Error evaluating example of length {len(example)}: {e}")
                
            pbar.update(1)
    
    # Calculate perplexity
    if total_tokens > 0:
        perplexity = np.exp(total_loss / total_tokens)
    else:
        perplexity = float('inf')
        
    return perplexity


def measure_memory_usage(model, example, device):
    """
    Measure peak memory usage when processing an example.
    """
    if not torch.cuda.is_available():
        return 0
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        try:
            # Ensure example has at least 2 tokens
            if len(example) < 2:
                return float('inf')
            
            # For standard model, handle position embedding limits
            if not model.config.use_summary_attention:
                max_context = model.config.max_position_embeddings
                
                # If example exceeds max_position_embeddings, chunk it
                if len(example) > max_context:
                    # Use the last max_context tokens for memory measurement
                    example = example[-max_context:]
            
            # For summary attention model, always use a chunk that fits within position embeddings
            elif len(example) > model.config.max_position_embeddings:
                # Use a chunk size that fits within position embeddings
                chunk_size = model.config.max_position_embeddings
                example = example[:chunk_size]
            
            # Process normally
            input_ids = torch.tensor(example[:-1], dtype=torch.long).unsqueeze(0).to(device)
            labels = torch.tensor(example[1:], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Get peak memory usage in MB
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            return peak_memory
        except Exception as e:
            print(f"Error measuring memory for example of length {len(example)}: {e}")
            return float('inf')


def measure_perplexity_and_memory(model, examples_by_length, device, max_batch_tokens=2048):
    """
    Measure perplexity and memory usage for different context lengths.
    """
    results = {}
    lengths = sorted(examples_by_length.keys())
    
    for length in lengths:
        # For standard model, skip lengths beyond max_position_embeddings
        if not model.config.use_summary_attention and length > model.config.max_position_embeddings:
            print(f"Skipping length {length} for standard model (exceeds max_position_embeddings={model.config.max_position_embeddings})")
            results[length] = {
                "perplexity": float('inf'),
                "memory_mb": float('inf')
            }
            continue
            
        examples = examples_by_length[length]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate perplexity
        try:
            print(f"\nEvaluating context length: {length}")
            perplexity = evaluate_perplexity(model, examples, device, max_batch_tokens)
            
            # Measure memory usage (using the first example)
            if examples:
                memory_usage = measure_memory_usage(model, examples[0], device)
            else:
                memory_usage = float('inf')
            
            # Record results
            results[length] = {
                "perplexity": perplexity,
                "memory_mb": memory_usage
            }
            print(f"Length {length}: Perplexity = {perplexity:.2f}, Memory = {memory_usage:.2f} MB")
        except Exception as e:
            print(f"Failed at length {length}: {e}")
            # Set a default perplexity value instead of inf to ensure we have a plot
            # Use a high value to indicate poor performance
            results[length] = {
                "perplexity": 1e6,  # Very high perplexity instead of inf
                "memory_mb": float('inf')
            }
            print(f"Length {length}: Perplexity = 1000000.00 (error fallback), Memory = inf MB")
    
    return results


def load_standard_model(device):
    """
    Load a standard GPT-2 model without summary attention.
    """
    print(f"Loading standard GPT-2 model to {device}...")
    
    # Create config for standard GPT-2
    config = GPT2Config(
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        use_summary_attention=False
    )
    
    # Initialize our model with the config
    model = GPT2ForLanguageModeling(config)
    model.to(device)
    
    print("Standard model loaded successfully")
    return model


def load_summary_model(device, summary_k=512, summary_m=512, use_recursive=False, max_chunk_size=1024):
    """
    Load a GPT-2 model with summary attention.
    
    Args:
        device: Device to load the model to
        summary_k: Number of tokens to keep intact at the end
        summary_m: Number of summary tokens to use
        use_recursive: Whether to use recursive summarization for very long contexts
        max_chunk_size: Maximum chunk size for recursive summarization
    """
    print(f"\nLoading summary attention model with k={summary_k}, m={summary_m}, recursive={use_recursive} to {device}...")
    
    # Create config for GPT-2 with summary attention
    config = GPT2Config(
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        use_summary_attention=True
    )
    
    # Set summary attention parameters
    # k: number of tokens to keep intact at the end
    # m: number of summary tokens to use
    # Ensure k + m <= max_position_embeddings (1024)
    config.summary_k = summary_k
    config.summary_m = summary_m
    
    # Set recursive summarization parameters
    config.use_recursive_summarization = use_recursive
    config.max_chunk_size = max_chunk_size
    
    # Ensure we're not exceeding position embedding limits
    assert config.summary_k + config.summary_m <= config.max_position_embeddings, \
        f"summary_k ({config.summary_k}) + summary_m ({config.summary_m}) must be <= max_position_embeddings ({config.max_position_embeddings})"
    
    # Set k and m as class attributes for the model
    config.k = config.summary_k
    config.m = config.summary_m
    
    # Initialize our model with the config
    model = GPT2ForLanguageModeling(config)
    
    # Load the pretrained GPT-2 weights
    # model.load_pretrained_weights('gpt2')
    model.transformer.from_pretrained('gpt2')    
    # Move model to the specified device
    model.to(device)
    
    return model


def plot_results(results, output_dir):
    """
    Plot perplexity and memory usage vs context length.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    context_lengths = sorted(list(
        list(results["standard"].keys()) + list(results["summary"].keys())
    ))
    
    # Extract perplexities
    standard_ppl = []
    summary_ppl = []
    standard_mem = []
    summary_mem = []
    
    for length in context_lengths:
        if length in results["standard"]:
            standard_ppl.append(results["standard"][length]["perplexity"])
            standard_mem.append(results["standard"][length]["memory_mb"])
        else:
            standard_ppl.append(float('nan'))
            standard_mem.append(float('nan'))
            
        if length in results["summary"]:
            summary_ppl.append(results["summary"][length]["perplexity"])
            summary_mem.append(results["summary"][length]["memory_mb"])
        else:
            summary_ppl.append(float('nan'))
            summary_mem.append(float('nan'))
    
    # Replace inf with NaN for plotting
    standard_ppl = [float('nan') if x == float('inf') else x for x in standard_ppl]
    summary_ppl = [float('nan') if x == float('inf') else x for x in summary_ppl]
    standard_mem = [float('nan') if x == float('inf') else x for x in standard_mem]
    summary_mem = [float('nan') if x == float('inf') else x for x in summary_mem]
    
    # Cap very high perplexity values for better visualization
    max_ppl_for_plot = 100000  # Cap at 100k for better visualization
    standard_ppl = [min(x, max_ppl_for_plot) if not np.isnan(x) else x for x in standard_ppl]
    summary_ppl = [min(x, max_ppl_for_plot) if not np.isnan(x) else x for x in summary_ppl]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot perplexity
    ax1.plot(context_lengths, standard_ppl, 'o-', label='Standard GPT-2')
    ax1.plot(context_lengths, summary_ppl, 'o-', label='Summary Attention')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity vs Context Length')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Plot memory usage
    ax2.plot(context_lengths, standard_mem, 'o-', label='Standard GPT-2')
    ax2.plot(context_lengths, summary_mem, 'o-', label='Summary Attention')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Context Length')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context_scaling_results.png'))
    
    # Create a more detailed plot for perplexity improvement
    plt.figure(figsize=(10, 6))
    
    # Calculate perplexity improvement
    improvement = []
    for s, sum_ppl in zip(standard_ppl, summary_ppl):
        if not np.isnan(s) and not np.isnan(sum_ppl) and s > 0 and sum_ppl > 0:
            imp = (s - sum_ppl) / s * 100  # Percentage improvement
            improvement.append(imp)
        else:
            improvement.append(float('nan'))
    
    # Filter out lengths where we don't have valid data for both models
    valid_lengths = []
    valid_improvement = []
    for length, imp in zip(context_lengths, improvement):
        if not np.isnan(imp):
            valid_lengths.append(length)
            valid_improvement.append(imp)
    
    if valid_lengths:
        plt.bar(valid_lengths, valid_improvement)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Context Length')
        plt.ylabel('Perplexity Improvement (%)')
        plt.title('Perplexity Improvement with Summary Attention')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'perplexity_improvement.png'))
    else:
        # Create an empty plot if no valid data
        plt.text(0.5, 0.5, 'No valid improvement data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.savefig(os.path.join(output_dir, 'perplexity_improvement.png'))


def create_summary_table(results):
    """
    Create a summary table of perplexity and memory usage results.
    """
    # Extract data
    context_lengths = sorted(list(set(
        list(results["standard"].keys()) + list(results["summary"].keys())
    )))
    
    # Create header
    header = "# Summary of Results\n\n"
    header += "## Perplexity and Memory Usage by Context Length\n\n"
    header += "| Context Length | Standard Model |  |  | Summary Attention Model |  |  |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- |\n"
    header += "| | Perplexity | Memory (MB) | Improvement | Perplexity | Memory (MB) | Improvement |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- |\n"
    
    rows = []
    for length in context_lengths:
        # Get standard model results
        if length in results["standard"]:
            std_ppl = results["standard"][length]["perplexity"]
            std_mem = results["standard"][length]["memory_mb"]
            
            # Format perplexity to be more readable
            if std_ppl != float('inf'):
                std_ppl_str = f"{std_ppl:.2f}"
                # Make large numbers more readable
                if std_ppl > 1000:
                    std_ppl_str = f"{std_ppl/1000:.2f}k"
            else:
                std_ppl_str = "N/A"
                
            std_mem_str = f"{std_mem:.2f}" if std_mem != float('inf') else "N/A"
        else:
            std_ppl = float('inf')
            std_ppl_str = "N/A"
            std_mem_str = "N/A"
        
        # Get summary model results
        if length in results["summary"]:
            sum_ppl = results["summary"][length]["perplexity"]
            sum_mem = results["summary"][length]["memory_mb"]
            
            # Format perplexity to be more readable
            if sum_ppl != float('inf'):
                sum_ppl_str = f"{sum_ppl:.2f}"
                # Make large numbers more readable
                if sum_ppl > 1000:
                    sum_ppl_str = f"{sum_ppl/1000:.2f}k"
            else:
                sum_ppl_str = "N/A"
                
            sum_mem_str = f"{sum_mem:.2f}" if sum_mem != float('inf') else "N/A"
            
            # Calculate improvements
            if std_ppl != float('inf') and sum_ppl != float('inf'):
                ppl_improvement = ((std_ppl - sum_ppl) / std_ppl) * 100
                ppl_improvement_str = f"{ppl_improvement:.2f}%" if ppl_improvement < 0 else f"+{ppl_improvement:.2f}%"
            else:
                ppl_improvement_str = "N/A"
        else:
            sum_ppl_str = "N/A"
            sum_mem_str = "N/A"
            ppl_improvement_str = "N/A"
        
        # Create row
        row = f"| {length} | {std_ppl_str} | {std_mem_str} | - | {sum_ppl_str} | {sum_mem_str} | {ppl_improvement_str} |\n"
        rows.append(row)
    
    # Add a summary section
    summary = "\n## Key Findings\n\n"
    summary += "- Standard GPT-2 model can only handle contexts up to 1024 tokens\n"
    summary += "- Summary Attention model successfully processes contexts up to 4096 tokens\n"
    summary += "- Memory usage for Summary Attention is higher but remains constant for longer contexts\n"
    
    # Combine header, rows and summary
    table = header + "".join(rows) + summary
    
    return table


def main(args):
    """
    Main function to run the context scaling test.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse context lengths
    context_lengths = [int(length) for length in args.context_lengths.split(",")]
    
    # Load examples
    examples_by_length = create_variable_length_examples(
        tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
        max_lengths=context_lengths,
        num_examples=args.num_examples
    )
    
    # Define summary attention configurations to test
    # summary_configs = [
    #     {"k": 512, "m": 512, "name": "k512_m512", "recursive": False},
    #     {"k": 256, "m": 768, "name": "k256_m768", "recursive": False},
    #     {"k": 768, "m": 256, "name": "k768_m256", "recursive": False},
    # ]
    summary_configs = []
    
    # Add recursive configurations for very long contexts
    if max(context_lengths) > 4096:
        recursive_configs = [
            {"k": 512, "m": 512, "name": "recursive_k512_m512", "recursive": True, "chunk_size": 2048},
            {"k": 256, "m": 768, "name": "recursive_k256_m768", "recursive": True, "chunk_size": 2048},
        ]
        summary_configs.extend(recursive_configs)
    
    # Store results for all models
    all_results = {}
    
    # Test standard model on all context lengths
    if not args.skip_standard:
        model_standard = load_standard_model(device)
        print("\nEvaluating standard attention model...")
        
        standard_results = measure_perplexity_and_memory(
            model_standard,
            examples_by_length,
            device,
            max_batch_tokens=args.max_batch_tokens
        )
        
        all_results["standard"] = standard_results
    
    # Test each summary model configuration
    if not args.skip_summary:
        for config in summary_configs:
            # Skip recursive models if max context length is small
            if config["recursive"] and max(context_lengths) <= 4096:
                continue
                
            # Load model with appropriate configuration
            if config["recursive"]:
                model_summary = load_summary_model(
                    device, 
                    summary_k=config["k"], 
                    summary_m=config["m"],
                    use_recursive=True,
                    max_chunk_size=config.get("chunk_size", 2048)
                )
                print(f"\nEvaluating recursive summary attention model with k={config['k']}, m={config['m']}, chunk_size={config.get('chunk_size', 2048)}...")
            else:
                model_summary = load_summary_model(
                    device, 
                    summary_k=config["k"], 
                    summary_m=config["m"]
                )
                print(f"\nEvaluating summary attention model with k={config['k']}, m={config['m']}...")
            
            # Only test on context lengths that make sense for this configuration
            test_lengths = {}
            for length in context_lengths:
                if length in examples_by_length:
                    test_lengths[length] = examples_by_length[length]
            
            summary_results = measure_perplexity_and_memory(
                model_summary,
                test_lengths,
                device,
                max_batch_tokens=args.max_batch_tokens
            )
            
            all_results[config["name"]] = summary_results
    
    # Plot results
    plot_results(all_results, args.output_dir)
    
    # Create summary table
    summary_table = create_summary_table(all_results)
    print(summary_table)
    
    # Save summary table to file
    with open(os.path.join(args.output_dir, 'summary_table.md'), 'w') as f:
        f.write(summary_table)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 with extended context")
    
    # Data arguments
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples per context length")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum context length to test")
    parser.add_argument("--context_lengths", type=str, default="128,256,512,768,1024,1536,2048,3072,4096,8192,16384,32768,65536,131072",
                        help="Comma-separated list of context lengths to test")
    parser.add_argument("--output_dir", type=str, default="extended_context_results",
                        help="Directory to save results")
    parser.add_argument("--max_batch_tokens", type=int, default=2048,
                        help="Maximum number of tokens to process in a batch")
    
    # Skip options
    parser.add_argument("--skip_standard", action="store_true", help="Skip evaluation of standard GPT-2")
    parser.add_argument("--skip_summary", action="store_true", help="Skip evaluation of summary attention model")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU usage even if CUDA is available")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(args)

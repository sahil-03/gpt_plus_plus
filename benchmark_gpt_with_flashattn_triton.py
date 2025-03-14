import torch
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Config
from models.gpt2 import GPT2Model
from modules.attention import CausalSelfAttention
from modules.flash_attention import FlashAttention
import gc

def measure_memory():
    """Measure current GPU memory usage"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB

def benchmark_model(model, input_ids, attention_mask, num_runs=2):
    """Benchmark model forward pass latency and memory usage"""
    # Move model to fp16 for better performance and compatibility with flash attention
    model.half()
    
    # Don't convert input_ids or attention_mask to fp16 as they need to be integers
    # for the embedding layer and indexing operations
    
    # Warm-up run
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Measure memory before inference
    memory_before = measure_memory()
    
    # Measure latency
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_runs * 1000  # Convert to ms
    
    # Measure peak memory during inference
    memory_after = measure_memory()
    
    return {
        'latency_ms': avg_latency,
        'memory_usage_mb': memory_after,
        'memory_increase_mb': memory_after - memory_before
    }

def create_gpt2_model(config, use_flash_attn=False):
    """Create GPT2 model with either standard or flash attention"""
    # Add missing attributes to the config
    if not hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = 0.1
    if not hasattr(config, 'attention_probs_dropout_prob'):
        config.attention_probs_dropout_prob = 0.1
    if not hasattr(config, 'layer_norm_eps'):
        config.layer_norm_eps = 1e-5
    if not hasattr(config, 'intermediate_size'):
        config.intermediate_size = config.hidden_size * 4
    if not hasattr(config, 'pad_token_id'):
        config.pad_token_id = 0
    
    # Create the model
    model = GPT2Model(config)
    model.cuda()
    model.eval()
    
    # # Restore the original attention implementation if needed
    # if use_flash_attn:
    #     CausalSelfAttention.__new__ = original_attn.__new__
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Benchmark GPT-2 with Flash Attention')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for each benchmark')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'xl'])
    args = parser.parse_args()
    
    # Define model configurations
    model_configs = {
        'small': GPT2Config(n_positions=1024, n_embd=768, n_layer=12, n_head=12, use_flash_attn=False),
        'medium': GPT2Config(n_positions=1024, n_embd=1024, n_layer=24, n_head=16, use_flash_attn=False),
        'large': GPT2Config(n_positions=1024, n_embd=1280, n_layer=36, n_head=20, use_flash_attn=False),
        'xl': GPT2Config(n_positions=1024, n_embd=1600, n_layer=48, n_head=25, use_flash_attn=False)
    }
    
    config = model_configs[args.model_size]
    
    # Results storage
    results = {
        'standard': {'latency': {}, 'memory': {}},
        'flash': {'latency': {}, 'memory': {}}
    }
    
    # Run benchmarks for different batch sizes and sequence lengths
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lengths:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
            
            # Create input tensors - use long for input_ids as they are token indices
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda', dtype=torch.long)
            attention_mask = torch.ones((batch_size, seq_len), device='cuda', dtype=torch.long)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Benchmark standard attention
            print("Testing standard attention...")
            config.use_flash_attn = False
            standard_model = create_gpt2_model(config, use_flash_attn=False)
            standard_metrics = benchmark_model(standard_model, input_ids, attention_mask, args.num_runs)
            
            print(f"Standard Attention - Latency: {standard_metrics['latency_ms']:.2f} ms, "
                  f"Memory: {standard_metrics['memory_usage_mb']:.2f} MB")
            
            # Clear GPU memory
            del standard_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Benchmark flash attention
            print("Testing flash attention...")
            config.use_flash_attn = True
            flash_model = create_gpt2_model(config, use_flash_attn=True)
            flash_metrics = benchmark_model(flash_model, input_ids, attention_mask, args.num_runs)
            
            print(f"Flash Attention - Latency: {flash_metrics['latency_ms']:.2f} ms, "
                  f"Memory: {flash_metrics['memory_usage_mb']:.2f} MB")
            
            # Calculate speedup and memory savings
            speedup = standard_metrics['latency_ms'] / flash_metrics['latency_ms']
            memory_reduction = 1 - (flash_metrics['memory_usage_mb'] / standard_metrics['memory_usage_mb'])
            
            print(f"Speedup: {speedup:.2f}x, Memory reduction: {memory_reduction*100:.2f}%")
            
            # Store results
            key = f"{batch_size}_{seq_len}"
            results['standard']['latency'][key] = standard_metrics['latency_ms']
            results['standard']['memory'][key] = standard_metrics['memory_usage_mb']
            results['flash']['latency'][key] = flash_metrics['latency_ms']
            results['flash']['memory'][key] = flash_metrics['memory_usage_mb']
            
            # Clear GPU memory
            del flash_model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Plot results
    plot_results(results, args.batch_sizes, args.seq_lengths)

def plot_results(results, batch_sizes, seq_lengths):
    """Plot benchmark results"""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot latency comparison for different batch sizes (fixed sequence length)
    seq_len = seq_lengths[len(seq_lengths) // 2]  # Use middle sequence length
    batch_labels = [str(b) for b in batch_sizes]
    standard_latency = [results['standard']['latency'][f"{b}_{seq_len}"] for b in batch_sizes]
    flash_latency = [results['flash']['latency'][f"{b}_{seq_len}"] for b in batch_sizes]
    
    axs[0, 0].bar(np.arange(len(batch_sizes)) - 0.2, standard_latency, width=0.4, label='Standard Attention')
    axs[0, 0].bar(np.arange(len(batch_sizes)) + 0.2, flash_latency, width=0.4, label='Flash Attention')
    axs[0, 0].set_xticks(np.arange(len(batch_sizes)))
    axs[0, 0].set_xticklabels(batch_labels)
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Latency (ms)')
    axs[0, 0].set_title(f'Latency vs Batch Size (Seq Len = {seq_len})')
    axs[0, 0].legend()
    
    # Plot memory usage comparison for different batch sizes
    standard_memory = [results['standard']['memory'][f"{b}_{seq_len}"] for b in batch_sizes]
    flash_memory = [results['flash']['memory'][f"{b}_{seq_len}"] for b in batch_sizes]
    
    axs[0, 1].bar(np.arange(len(batch_sizes)) - 0.2, standard_memory, width=0.4, label='Standard Attention')
    axs[0, 1].bar(np.arange(len(batch_sizes)) + 0.2, flash_memory, width=0.4, label='Flash Attention')
    axs[0, 1].set_xticks(np.arange(len(batch_sizes)))
    axs[0, 1].set_xticklabels(batch_labels)
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Memory Usage (MB)')
    axs[0, 1].set_title(f'Memory Usage vs Batch Size (Seq Len = {seq_len})')
    axs[0, 1].legend()
    
    # Plot latency comparison for different sequence lengths (fixed batch size)
    batch_size = batch_sizes[len(batch_sizes) // 2]  # Use middle batch size
    seq_labels = [str(s) for s in seq_lengths]
    standard_latency = [results['standard']['latency'][f"{batch_size}_{s}"] for s in seq_lengths]
    flash_latency = [results['flash']['latency'][f"{batch_size}_{s}"] for s in seq_lengths]
    
    axs[1, 0].bar(np.arange(len(seq_lengths)) - 0.2, standard_latency, width=0.4, label='Standard Attention')
    axs[1, 0].bar(np.arange(len(seq_lengths)) + 0.2, flash_latency, width=0.4, label='Flash Attention')
    axs[1, 0].set_xticks(np.arange(len(seq_lengths)))
    axs[1, 0].set_xticklabels(seq_labels)
    axs[1, 0].set_xlabel('Sequence Length')
    axs[1, 0].set_ylabel('Latency (ms)')
    axs[1, 0].set_title(f'Latency vs Sequence Length (Batch Size = {batch_size})')
    axs[1, 0].legend()
    
    # Plot memory usage comparison for different sequence lengths
    standard_memory = [results['standard']['memory'][f"{batch_size}_{s}"] for s in seq_lengths]
    flash_memory = [results['flash']['memory'][f"{batch_size}_{s}"] for s in seq_lengths]
    
    axs[1, 1].bar(np.arange(len(seq_lengths)) - 0.2, standard_memory, width=0.4, label='Standard Attention')
    axs[1, 1].bar(np.arange(len(seq_lengths)) + 0.2, flash_memory, width=0.4, label='Flash Attention')
    axs[1, 1].set_xticks(np.arange(len(seq_lengths)))
    axs[1, 1].set_xticklabels(seq_labels)
    axs[1, 1].set_xlabel('Sequence Length')
    axs[1, 1].set_ylabel('Memory Usage (MB)')
    axs[1, 1].set_title(f'Memory Usage vs Sequence Length (Batch Size = {batch_size})')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('gpt2_flash_attention_benchmark.png')
    plt.close()
    print("Results plotted and saved to 'gpt2_flash_attention_benchmark.png'")

if __name__ == "__main__":
    main()
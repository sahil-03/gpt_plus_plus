import matplotlib.pyplot as plt
import re
import numpy as np
import os

def parse_log_file(file_path):
    """Parse the log file to extract batch numbers and loss values"""
    batch_numbers = []
    losses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Use regex to extract batch number and loss
            match = re.search(r'Batch (\d+), Loss: (\d+\.\d+)', line)
            if match:
                batch_number = int(match.group(1))
                loss = float(match.group(2))
                batch_numbers.append(batch_number)
                losses.append(loss)
    
    return batch_numbers, losses

# File paths for the three experiments
file_paths = [
    'training_sonnet_adam_20250314_200525.log',
    'training_sonnet_preconditioned_adam_20250314_200800.log',
    'training_sonnet_preconditioner_20250314_201108.log'
]

# Labels for the different optimization methods
method_labels = [
    'Adam',
    'PreconditionedAdam',
    'Preconditioner',
]

# Set up the plot
plt.figure(figsize=(10, 6))

# Colors for the different methods
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

method_to_info = {}
# Plot each experiment
for i, (file_path, label) in enumerate(zip(file_paths, method_labels)):
    try:
        batch_numbers, losses = parse_log_file(file_path)
        method_to_info[label] = (batch_numbers, losses)
        plt.plot(batch_numbers, losses, marker='o', linestyle='-', 
                 label=label, color=colors[i], markersize=4)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Add a vertical line every 12 batches (assuming each epoch is 12 batches)
# Uncomment if this applies to your data
# for epoch in range(1, max(batch_numbers) // 12 + 1):
#     plt.axvline(x=epoch * 12, color='gray', linestyle='--', alpha=0.3)

# Customize the plot
plt.xlabel('Batch Number', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Sonnet Generation: Training Loss of Different Optimizers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Set y-axis to start from 0 or slightly below the minimum loss for better visualization
min_loss = min([min(losses[1]) for losses in method_to_info.values()])
plt.ylim(min_loss * 0.9, None)
print("here")

# Add annotations for the final loss values
for i, (file_path, label) in enumerate(zip(file_paths, method_labels)):
    batch_numbers, losses = method_to_info[label]
    try:
        if losses:
         plt.annotate(f'{losses[-1]:.2f}', 
                        xy=(batch_numbers[-1], losses[-1]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        color=colors[i],
                        fontweight='bold')
    except Exception as e:
        print(f"Error annotating {label}: {e}")

plt.tight_layout()

# Save the figure - now with explicit path
output_path = os.path.join(os.getcwd(), 'optimization_comparison.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")

# Optional: Show the plot if in an interactive environment
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot (this is normal in non-interactive environments): {e}")

print("Script completed.")
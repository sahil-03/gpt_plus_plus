'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import logging
from datetime import datetime
import os

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW, DiagonalPreconditioner, PreconditionedAdam

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  
def setup_logging(args):
  log_dir = "logs"
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  logging.basicConfig(
    filename=os.path.join(log_dir, f"training_sonnet_{args.optimizer}_{timestamp}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
  )

class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Freeze the base model parameters to preserve pre-trained knowledge
    for param in self.gpt.parameters():
      param.requires_grad = False
      
    # 1. The final layer norm - important for output distribution
    for param in self.gpt.final_layer_norm.parameters():
      param.requires_grad = True
      
    # 2. The last few transformer layers - these capture higher-level features
    num_layers_to_finetune = min(4, len(self.gpt.gpt_layers))  # Fine-tune at most 4 layers
    for i in range(len(self.gpt.gpt_layers) - num_layers_to_finetune, len(self.gpt.gpt_layers)):
      for param in self.gpt.gpt_layers[i].parameters():
        param.requires_grad = True
        
    # 3. Task-specific adapter layer for sonnet generation
    self.sonnet_adapter = nn.Linear(args.d, args.d)
    self.adapter_activation = nn.GELU()

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs['last_hidden_state'] 
    adapted_states = self.adapter_activation(self.sonnet_adapter(hidden_states))
    hidden_states = hidden_states + adapted_states
    logits = torch.matmul(hidden_states, self.gpt.word_embedding.weight.transpose(0, 1))
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    
    line_count = 0
    max_lines = 14  # Shakesperean sonnets have max of 14 lines
    newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
    
    top_k = 50  

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)
      
      # Top-k filtering
      top_k_probs, top_k_indices = torch.topk(probs, top_k)
      
      # Top-p (nucleus) sampling on the reduced top-k set
      sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      
      # Create top-p mask
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  
      top_p_mask[..., 0] = True 
      
      # Apply the mask
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
      
      # Map back to original indices
      filtered_indices = top_k_indices.gather(-1, sorted_indices)
      
      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = filtered_indices.gather(dim=-1, index=sampled_index)
      
      # Check for line endings to enforce sonnet structure
      if sampled_token.item() == newline_token_id:
        line_count += 1
        
        # If we've reached the desired number of lines, end generation
        if line_count >= max_lines:
          break
      
      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  
  if args.optimizer == 'adam':
    optimizer = AdamW(model.parameters(), lr=lr)
  elif args.optimizer == 'preconditioner':
    optimizer = DiagonalPreconditioner(model.parameters(), lr=lr)
  elif args.optimizer == 'preconditioned_adam':
    optimizer = PreconditionedAdam(model.parameters(), lr=lr)
  else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")
  
  # Early stopping parameters
  patience = 3 
  best_loss = float('inf')
  patience_counter = 0
  
  # Validation split for early stopping
  train_size = int(0.9 * len(sonnet_dataset))
  val_size = len(sonnet_dataset) - train_size
  train_subset, val_subset = torch.utils.data.random_split(sonnet_dataset, [train_size, val_size])
  
  train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size,
                               collate_fn=sonnet_dataset.collate_fn)
  val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=args.batch_size,
                             collate_fn=sonnet_dataset.collate_fn)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
      
      total_batches = epoch * len(train_dataloader) + num_batches
      logging.info(f"Batch {total_batches}, Loss: {loss.item():.4f}")

    train_loss = train_loss / num_batches
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
      for batch in tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        
        val_loss += loss.item()
        val_batches += 1
    
    val_loss = val_loss / val_batches
    
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, val loss :: {val_loss :.3f}")
    
    # Check for early stopping
    if val_loss < best_loss:
      best_loss = val_loss
      patience_counter = 0
      # Save the best model
      save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break
    
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--optimizer", type=str, default="adam",
                      choices=['adam', 'preconditioner', 'preconditioned_adam'],
                      help='Optimizer to use for training')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-{args.optimizer}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  setup_logging(args)
  train(args)
  generate_submission_sonnets(args)
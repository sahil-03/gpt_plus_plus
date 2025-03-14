import torch 
import torch.nn as nn 
from modules.attention import CausalSelfAttention
from transformers import GPT2Model, GPT2Config


class SummaryAttention(nn.Module):
    def __init__(self, config, m, k):
        super().__init__()

        self.m = m # number of summary tokens 
        self.k = k # number of tokens kept intact 
        self.hidden_size = config.hidden_size
        
        # New parameter for recursive summarization
        self.use_recursive_summarization = getattr(config, 'use_recursive_summarization', False)
        self.max_chunk_size = getattr(config, 'max_chunk_size', 1024)  # Maximum size of each chunk
        self.recursive_depth = getattr(config, 'recursive_depth', 0)   # Track recursion depth for debugging

        # Initialize a small GPT-2 model for summarization
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=1024,  # Standard GPT-2 position embedding size
            n_embd=config.hidden_size,
            n_layer=4,  # Using a smaller model for efficiency
            n_head=config.num_attention_heads,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=config.layer_norm_eps
        )
        
        # Load a pretrained GPT-2 model for summarization
        self.summarizer = GPT2Model.from_pretrained('gpt2')
        
        # Ensure the hidden size matches our model's hidden size
        if self.summarizer.config.n_embd != self.hidden_size:
            self.hidden_proj = nn.Linear(self.summarizer.config.n_embd, self.hidden_size)
        else:
            self.hidden_proj = nn.Identity()
            
        # Projection layer to generate exactly m summary tokens
        self.summary_projector = nn.Linear(self.hidden_size, self.m * self.hidden_size)
        
        self.attention = CausalSelfAttention(config)
        
        # print(f"Using summary attention with: m = {m}, k = {k}")
    
    def summarize_chunk(self, chunk, attention_mask=None, target_size=None):
        """
        Summarize a single chunk of tokens.
        
        Args:
            chunk: Tensor of shape [batch_size, chunk_size, hidden_size]
            attention_mask: Optional attention mask for the chunk
            target_size: Number of summary tokens to generate (defaults to self.m)
            
        Returns:
            Tensor of shape [batch_size, target_size, hidden_size]
        """
        batch_size, chunk_size, hidden_size = chunk.size()
        
        # Default to self.m if target_size not specified
        if target_size is None:
            target_size = self.m
            
        # Process the chunk with the summarizer
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, chunk_size, device=chunk.device)
            
        gpt2_outputs = self.summarizer(
            inputs_embeds=chunk,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the last hidden state from GPT-2
        gpt2_hidden_states = gpt2_outputs.last_hidden_state
        
        # Project to our hidden size if needed
        gpt2_hidden_states = self.hidden_proj(gpt2_hidden_states)
        
        # Use the last token's representation as a context vector
        context_vector = gpt2_hidden_states[:, -1, :]
        
        # Project the context vector to generate target_size summary tokens
        # We need to adjust the projection to match the target size
        if target_size == self.m:
            # Use the existing projector
            summary_tokens_flat = self.summary_projector(context_vector)
        else:
            # Create a temporary projector for the specific target size
            temp_projector = nn.Linear(self.hidden_size, target_size * hidden_size, 
                                      device=context_vector.device)
            summary_tokens_flat = temp_projector(context_vector)
        
        # Reshape to [batch_size, target_size, hidden_size]
        summary = summary_tokens_flat.view(batch_size, target_size, hidden_size)
        
        return summary
    
    def recursive_summarize(self, tokens_to_compress, attention_mask=None, target_size=None, depth=0):
        """
        Recursively summarize a sequence of tokens that may be longer than max_chunk_size.
        
        Args:
            tokens_to_compress: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            target_size: Number of summary tokens to generate (defaults to self.m)
            depth: Current recursion depth
            
        Returns:
            Tensor of shape [batch_size, target_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = tokens_to_compress.size()
        
        # Default to self.m if target_size not specified
        if target_size is None:
            target_size = self.m
            
        # Base case: if tokens fit within max_chunk_size, summarize directly
        if seq_len <= self.max_chunk_size:
            return self.summarize_chunk(tokens_to_compress, attention_mask, target_size)
        
        # Split into chunks
        num_chunks = (seq_len + self.max_chunk_size - 1) // self.max_chunk_size
        chunks = []
        chunk_masks = []
        
        for i in range(num_chunks):
            start_idx = i * self.max_chunk_size
            end_idx = min((i + 1) * self.max_chunk_size, seq_len)
            
            # Extract chunk
            chunk = tokens_to_compress[:, start_idx:end_idx, :]
            chunks.append(chunk)
            
            # Extract corresponding attention mask if provided
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start_idx:end_idx]
                chunk_masks.append(chunk_mask)
        
        # Calculate tokens per chunk summary
        tokens_per_summary = max(1, target_size // num_chunks)
        remainder = target_size % num_chunks
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Adjust target size to account for remainder
            chunk_target_size = tokens_per_summary + (1 if i < remainder else 0)
            
            # Get corresponding mask if available
            chunk_mask = chunk_masks[i] if chunk_masks else None
            
            # Summarize the chunk
            summary = self.summarize_chunk(chunk, chunk_mask, chunk_target_size)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summaries = torch.cat(chunk_summaries, dim=1)
        
        # If combined summaries are still too large, recurse
        if combined_summaries.size(1) > self.max_chunk_size:
            # Create a new attention mask for the combined summaries
            combined_mask = torch.ones(batch_size, combined_summaries.size(1), device=combined_summaries.device)
            
            # Recursive call with increased depth
            return self.recursive_summarize(combined_summaries, combined_mask, target_size, depth + 1)
        
        # If we need to further compress the combined summaries
        if combined_summaries.size(1) > target_size:
            return self.summarize_chunk(combined_summaries, None, target_size)
        
        return combined_summaries
    
    def forward(self, hidden_states, attention_mask=None, **kwargs): 
        batch_size, n, d = hidden_states.size() 
        
        # Use the fixed values of k and m from config
        # Don't dynamically adjust them based on sequence length
        k = self.k
        m = self.m
        
        # For very small sequences, just use standard attention
        if n <= 384:  # This threshold can be adjusted
            return self.attention(hidden_states, attention_mask, **kwargs)
        
        # For sequences longer than max_position_embeddings, we need to compress
        # Partition tokens - keep the last k tokens intact, compress the rest
        if n > k:
            tokens_to_compress = hidden_states[:, :n-k, :]
            tokens_to_keep = hidden_states[:, n-k:, :]
        else:
            # If sequence is shorter than k, keep all tokens
            tokens_to_compress = torch.zeros((batch_size, 0, d), device=hidden_states.device)
            tokens_to_keep = hidden_states
        
        # Only compress if we have tokens to compress
        if tokens_to_compress.size(1) > 0:
            # Create attention mask for the GPT-2 model
            # GPT-2 uses 1 for tokens to attend to and 0 for tokens to ignore
            if attention_mask is not None:
                # Extract the part of the attention mask corresponding to tokens_to_compress
                # Only if there are tokens to compress
                if tokens_to_compress.size(1) > 0 and attention_mask.size(3) >= n-k:
                    gpt2_attention_mask = attention_mask[:, :, :, :n-k]
                    # Convert from 0/-10000 format to 0/1 format expected by GPT-2
                    gpt2_attention_mask = (gpt2_attention_mask == 0).long()
                    # Reshape to [batch_size, seq_len]
                    gpt2_attention_mask = gpt2_attention_mask.squeeze(1).squeeze(1)
                else:
                    gpt2_attention_mask = torch.ones(batch_size, tokens_to_compress.size(1), device=hidden_states.device)
            else:
                gpt2_attention_mask = torch.ones(batch_size, tokens_to_compress.size(1), device=hidden_states.device)
            
            # Choose between recursive and standard summarization
            if self.use_recursive_summarization and tokens_to_compress.size(1) > self.max_chunk_size:
                # Use recursive summarization for very long sequences
                summary = self.recursive_summarize(tokens_to_compress, gpt2_attention_mask, self.m)
                
                # Log information about recursive summarization
                print(f"Used recursive summarization for sequence of length {tokens_to_compress.size(1)}")
            else:
                # Handle sequences longer than GPT-2's max position embeddings (1024)
                if tokens_to_compress.size(1) > 1024:
                    # Process in chunks of 1024 tokens
                    chunk_size = 1024
                    last_chunk_start = max(0, tokens_to_compress.size(1) - chunk_size)
                    last_chunk = tokens_to_compress[:, last_chunk_start:, :]
                    
                    # Make sure we have a valid mask for the last chunk
                    if last_chunk_start < gpt2_attention_mask.size(1):
                        last_chunk_mask = gpt2_attention_mask[:, last_chunk_start:]
                    else:
                        last_chunk_mask = torch.ones(batch_size, last_chunk.size(1), device=hidden_states.device)
                    
                    gpt2_outputs = self.summarizer(
                        inputs_embeds=last_chunk,
                        attention_mask=last_chunk_mask,
                        return_dict=True
                    )
                    
                    # Get the last hidden state from GPT-2
                    gpt2_hidden_states = gpt2_outputs.last_hidden_state
                else:
                    # Process normally if within position embedding limit
                    gpt2_outputs = self.summarizer(
                        inputs_embeds=tokens_to_compress,
                        attention_mask=gpt2_attention_mask,
                        return_dict=True
                    )
                    
                    # Get the last hidden state from GPT-2
                    gpt2_hidden_states = gpt2_outputs.last_hidden_state
                
                # Project to our hidden size if needed
                gpt2_hidden_states = self.hidden_proj(gpt2_hidden_states)
                
                # Use the last token's representation as a context vector
                context_vector = gpt2_hidden_states[:, -1, :]
                
                # Project the context vector to generate m summary tokens
                summary_tokens_flat = self.summary_projector(context_vector)
                
                # Reshape to [batch_size, m, hidden_size]
                summary = summary_tokens_flat.view(batch_size, m, d)
        else:
            # No tokens to compress, create empty summary
            summary = torch.zeros((batch_size, 0, d), device=hidden_states.device)
        
        # Concatenate summary tokens with tokens to keep
        compressed_hidden = torch.cat([summary, tokens_to_keep], dim=1)
        
        # Adjust attention mask for the new sequence length
        if attention_mask is not None:
            # Create mask for summary tokens (allow attending to all summary tokens)
            summary_mask = torch.ones((batch_size, 1, 1, summary.size(1)), device=attention_mask.device)
            
            # Keep the mask for tokens_to_keep
            if n > k and attention_mask.size(3) >= n:
                keep_mask = attention_mask[:, :, :, n-k:]
            else:
                # Create a default mask if we can't extract from the original
                keep_mask = torch.ones((batch_size, 1, 1, tokens_to_keep.size(1)), device=attention_mask.device)
            
            # Combine masks
            new_attention_mask = torch.cat([summary_mask, keep_mask], dim=-1)
        else:
            new_attention_mask = None
        
        # Pass summarized sequence through attention
        compressed_output = self.attention(compressed_hidden, new_attention_mask, **kwargs)
        
        # Expand the summary tokens back to original sequence length
        if summary.size(1) > 0 and tokens_to_compress.size(1) > 0:
            # Extract summary output
            summary_output = compressed_output[:, :summary.size(1), :]
            
            # Use a simple expansion approach - repeat each summary token to cover the original sequence
            tokens_per_summary = tokens_to_compress.size(1) // m if m > 0 else 0
            remainder = tokens_to_compress.size(1) % m if m > 0 else 0
            
            expanded_summary = []
            
            for i in range(m):
                # Adjust chunk size to account for remainder
                current_chunk_size = tokens_per_summary + (1 if i < remainder else 0)
                
                if current_chunk_size > 0:
                    # Repeat the summary token for the chunk
                    expanded_token = summary_output[:, i:i+1, :].expand(-1, current_chunk_size, -1)
                    expanded_summary.append(expanded_token)
            
            # Concatenate expanded tokens
            if expanded_summary:
                expanded_output = torch.cat(expanded_summary, dim=1)
            else:
                # Fallback if no expanded tokens were created
                expanded_output = torch.zeros((batch_size, 0, d), device=hidden_states.device)
        else:
            expanded_output = torch.zeros((batch_size, 0, d), device=hidden_states.device)
        
        # Extract output for kept tokens
        if tokens_to_keep.size(1) > 0:
            kept_output = compressed_output[:, -tokens_to_keep.size(1):, :]
        else:
            kept_output = torch.zeros((batch_size, 0, d), device=hidden_states.device)
        
        # Combine expanded summary and kept output to match original sequence length
        attention_output = torch.cat([expanded_output, kept_output], dim=1)
        
        # Ensure output has the same shape as input
        if attention_output.size(1) != n:
            print(f"Warning: Output shape {attention_output.size()} doesn't match input shape {hidden_states.size()}")
            # Adjust output size if necessary
            if attention_output.size(1) < n:
                # Pad with zeros
                padding = torch.zeros((batch_size, n - attention_output.size(1), d), device=hidden_states.device)
                attention_output = torch.cat([attention_output, padding], dim=1)
            else:
                # Truncate
                attention_output = attention_output[:, :n, :]
        
        return attention_output
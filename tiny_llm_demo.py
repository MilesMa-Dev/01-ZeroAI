# tiny_llm_demo.py
# Enhanced Transformer LM with SentencePiece, RoPE, weight sharing, cosine LR, EMA, and top-p sampling
# Author: Miles + GPT-5 Thinking + Enhanced by Claude
# Run: python tiny_llm_demo.py

import math, sys, os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Optional, Tuple
from sp_tokenizer import DynamicSPTokenizer

# ------------------------------
# Config (enhanced with modern features)
# ------------------------------
initial_vocab_size = 500     # Dynamic SentencePiece starting vocab size
block_size   = 256         # context length (supports up to 2048 with NTK scaling)
n_embd       = 256           # model width
n_head       = 8             # attention heads
n_layer      = 4             # transformer layers
dropout      = 0.1           # dropout rate
device       = 'cuda' if torch.cuda.is_available() else 'cpu'

# Enhanced features config
use_rope     = True          # Rotary Position Embedding
tie_weights  = True          # Weight tying between embedding and output
use_ema      = True          # Exponential Moving Average
ema_decay    = 0.999         # EMA decay rate
top_p        = 0.9           # nucleus sampling
max_context  = 2048          # Maximum supported context length with NTK scaling

seed = 1337
torch.manual_seed(seed)
random.seed(seed)

# Global tokenizer instance - Dynamic SentencePiece starting with 500 vocab
tokenizer = DynamicSPTokenizer(base_vocab_size=500)

# ------------------------------
# Rotary Position Embedding (RoPE)
# ------------------------------
class RoPEEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base_freq: float = 10000.0, ntk_scaling: bool = True):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base_freq = base_freq
        self.ntk_scaling = ntk_scaling
        
        # Build initial RoPE cache
        self._build_rope_cache(max_seq_len)
    
    def _build_rope_cache(self, seq_len: int):
        """Build RoPE cache with optional NTK scaling"""
        # NTK scaling: adjust base frequency based on sequence length extension
        if self.ntk_scaling and seq_len > self.max_seq_len:
            # Calculate scaling factor for NTK
            scale_factor = seq_len / self.max_seq_len
            # Apply NTK scaling to base frequency
            effective_base_freq = self.base_freq * (scale_factor ** (self.dim / (self.dim - 2)))
            print(f"🔧 NTK scaling: seq_len={seq_len}, scale_factor={scale_factor:.2f}, effective_base_freq={effective_base_freq:.0f}")
        else:
            effective_base_freq = self.base_freq
        
        # Precompute frequencies with potentially scaled base
        freqs = 1.0 / (effective_base_freq ** (torch.arange(0, self.dim, 2).float() / self.dim))
        pos = torch.arange(seq_len).float()
        freqs_grid = torch.outer(pos, freqs)  # (seq_len, dim//2)
        
        self.register_buffer('freqs_cos', torch.cos(freqs_grid), persistent=False)
        self.register_buffer('freqs_sin', torch.sin(freqs_grid), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_heads, seq_len, head_dim)
        seq_len = x.size(2)
        
        # Check if we need to rebuild cache for longer sequences
        if seq_len > self.freqs_cos.size(0):
            print(f"🔧 Extending RoPE cache from {self.freqs_cos.size(0)} to {seq_len}")
            self._build_rope_cache(seq_len)
        
        # Get frequency components for current sequence length
        cos = self.freqs_cos[:seq_len]  # (seq_len, dim//2)
        sin = self.freqs_sin[:seq_len]  # (seq_len, dim//2)
        
        # Reshape cos/sin to match x dimensions: (1, 1, seq_len, dim//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Split x into even and odd dimensions
        x_even = x[..., ::2]   # (batch, n_heads, seq_len, head_dim//2)
        x_odd = x[..., 1::2]   # (batch, n_heads, seq_len, head_dim//2)
        
        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated

# ------------------------------
# Model: Enhanced GPT-like Transformer
# ------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.key    = nn.Linear(n_embd, n_embd, bias=False)
        self.query  = nn.Linear(n_embd, n_embd, bias=False)
        self.value  = nn.Linear(n_embd, n_embd, bias=False)
        self.proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        # RoPE embedding with extended support and NTK scaling
        if use_rope:
            self.rope = RoPEEmbedding(self.head_dim, max_seq_len=2048, ntk_scaling=True)
        else:
            self.rope = None
            
        # causal mask - make it larger to support extended contexts
        max_mask_size = 2048  # Support up to 2048 context length
        mask = torch.tril(torch.ones(max_mask_size, max_mask_size)).view(1, 1, max_mask_size, max_mask_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        
        # Compute queries, keys, values
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)  # (B, nh, T, hs)
        
        # Apply RoPE if enabled
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)
        
        # Reshape and project
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.initial_vocab_size = vocab_size
        self.tok_emb = nn.Embedding(max(vocab_size, 10), n_embd)  # Dynamic vocab expansion
        
        # Only use positional embedding if not using RoPE
        if not use_rope:
            self.pos_emb = nn.Embedding(block_size, n_embd)
        else:
            self.pos_emb = None
            
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, max(vocab_size, 10), bias=False)
        
        # Weight tying between embedding and output head
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def expand_vocab(self, new_vocab_size):
        """Expand vocabulary size if needed"""
        if new_vocab_size <= self.tok_emb.num_embeddings:
            return  # No expansion needed
        
        old_emb_size = self.tok_emb.num_embeddings
        old_head_size = self.head.out_features
        
        # Create new embedding layer with expanded vocabulary
        new_tok_emb = nn.Embedding(new_vocab_size, self.tok_emb.embedding_dim).to(self.tok_emb.weight.device)
        
        # Copy old weights
        with torch.no_grad():
            new_tok_emb.weight[:old_emb_size] = self.tok_emb.weight
            # Initialize new weights
            nn.init.normal_(new_tok_emb.weight[old_emb_size:], mean=0.0, std=0.02)
        
        self.tok_emb = new_tok_emb
        
        # Update head layer if not using weight tying
        if not tie_weights or self.head.weight.shape[0] != new_vocab_size:
            new_head = nn.Linear(self.head.in_features, new_vocab_size, bias=False).to(self.head.weight.device)
            
            with torch.no_grad():
                new_head.weight[:old_head_size] = self.head.weight
                # Initialize new weights
                nn.init.normal_(new_head.weight[old_head_size:], mean=0.0, std=0.02)
            
            self.head = new_head
            
            # Re-tie weights if needed
            if tie_weights:
                self.head.weight = self.tok_emb.weight
        
        # print(f"🔄 Expanded model vocabulary from {old_emb_size} to {new_vocab_size}")
    
    @property
    def current_vocab_size(self):
        """Get current vocabulary size"""
        return self.tok_emb.num_embeddings

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)                         # (B,T,C)
        
        # Add positional encoding if not using RoPE
        if self.pos_emb is not None:
            pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
            x = self.drop(tok + pos)
        else:
            x = self.drop(tok)
            
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                           # (B,T,vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None, adaptive_length=False):
        self.eval()
        
        # Ensure model vocabulary matches tokenizer
        tokenizer_vocab_size = tokenizer.vocab_size
        if self.current_vocab_size < tokenizer_vocab_size:
            print(f"🔧 Expanding model vocab from {self.current_vocab_size} to {tokenizer_vocab_size}")
            self.expand_vocab(tokenizer_vocab_size)
        
        # Adaptive length parameters
        if adaptive_length:
            min_length = max(10, max_new_tokens // 4)  # At least generate some content
            stop_chars = {'。', '.', '!', '！', '?', '？', '\n'}  # Natural stopping points
            repetition_window = 20  # Check for repetition in last N characters
        
        generated_length = 0
        for _ in range(max_new_tokens):
            # Check for adaptive stopping conditions
            if adaptive_length and generated_length >= min_length:
                # Convert current sequence to text for stopping condition checks
                current_text = tokenizer.decode(idx[0])
                
                # Stop at natural sentence endings
                if current_text and current_text[-1] in stop_chars:
                    print(f"🛑 Natural stop at sentence ending (length: {generated_length})")
                    break
                
                # Stop if we see repetitive patterns
                if len(current_text) >= repetition_window * 2:
                    recent = current_text[-repetition_window:]
                    prev = current_text[-repetition_window*2:-repetition_window]
                    if recent == prev:
                        print(f"🛑 Repetition detected, stopping (length: {generated_length})")
                        break
            
            generated_length += 1
            if idx.size(1) == 0:
                # For empty input, create a dummy input and use the first token prediction
                dummy_input = torch.zeros((1, 1), dtype=torch.long, device=idx.device)
                logits, _ = self(dummy_input)
                # Apply temperature with safety limits
                safe_temperature = max(0.1, min(temperature, 5.0))  # Limit temperature to [0.1, 5.0]
                logits = logits[:, -1, :] / safe_temperature
            else:
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                # Apply temperature with safety limits
                safe_temperature = max(0.1, min(temperature, 5.0))  # Limit temperature to [0.1, 5.0]
                logits = logits[:, -1, :] / safe_temperature
            
            # Get current vocabulary size first
            current_vocab_size = tokenizer.vocab_size
            model_vocab_size = logits.size(-1)
            
            # Apply top-k filtering (ensure k doesn't exceed vocabulary size)
            if top_k is not None:
                effective_vocab_size = min(current_vocab_size, model_vocab_size)
                effective_k = min(top_k, effective_vocab_size)
                if effective_k > 0:
                    v, _ = torch.topk(logits[..., :effective_vocab_size], effective_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
            probs = F.softmax(logits, dim=-1)
            
            # Handle edge case where all probabilities might be invalid
            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                # Fallback to uniform distribution over valid vocabulary
                probs = torch.ones_like(logits)
                probs = F.softmax(probs, dim=-1)
                # Update model_vocab_size after fallback
                model_vocab_size = probs.size(-1)
            
            # Ensure we don't sample beyond current vocabulary size
            
            if model_vocab_size > current_vocab_size:
                # Truncate to actual vocabulary size
                probs = probs[..., :current_vocab_size]
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            elif model_vocab_size < current_vocab_size:
                # Model hasn't been expanded yet, should not happen but handle it
                print(f"⚠️  Model vocab size ({model_vocab_size}) < tokenizer vocab size ({current_vocab_size})")
                current_vocab_size = model_vocab_size
            
            # Handle special tokens elimination based on vocabulary size
            if current_vocab_size > 2:
                probs[..., 0] = 0.0      # Completely eliminate PAD
                probs[..., 1] = 0.0      # Completely eliminate UNK  
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Safety check: if all probabilities are zero, use uniform over non-special tokens
                if torch.any(torch.isnan(probs)) or torch.any(probs.sum(dim=-1) == 0):
                    probs = torch.zeros_like(probs)
                    probs[..., 2:current_vocab_size] = 1.0 / max(1, current_vocab_size - 2)
            elif current_vocab_size == 2:
                # Special case: only PAD and UNK available, use UNK but warn
                probs[..., 0] = 0.0  # Eliminate PAD
                probs[..., 1] = 1.0  # Use UNK as fallback
                print("⚠️  Very small vocabulary, using UNK token")
            else:
                # Emergency fallback for vocab size 1
                probs[..., 0] = 1.0
                print("⚠️  Extremely small vocabulary, using PAD token")
            
            try:
                next_id = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                if "selected index k out of range" in str(e):
                    # Emergency fallback: choose a random valid token (avoid special tokens)
                    valid_range = min(current_vocab_size, probs.size(-1))
                    if valid_range > 2:  # Have at least PAD, UNK, and one more token
                        # Choose from non-special tokens (start from index 2)
                        next_id = torch.randint(2, valid_range, (probs.size(0), 1))
                    elif valid_range > 1:
                        # Use any available non-PAD token (even UNK is better than PAD)
                        next_id = torch.tensor([[1]], dtype=torch.long, device=probs.device)
                    else:
                        # Absolute fallback
                        next_id = torch.tensor([[1]], dtype=torch.long, device=probs.device)
                    print(f"🔧 Used fallback sampling: token {next_id.item()}")
                else:
                    raise e
            
            # Ensure generated token is within vocabulary bounds
            if next_id.item() >= current_vocab_size:
                # Choose a random valid non-special token instead of UNK
                if current_vocab_size > 2:
                    safe_id = torch.randint(2, current_vocab_size, (1,)).item()
                    next_id = torch.tensor([[safe_id]], dtype=torch.long, device=next_id.device)
                    print(f"🔧 Token out of bounds, using random valid token: {safe_id}")
                else:
                    next_id = torch.tensor([[tokenizer.unk_id]], dtype=torch.long, device=next_id.device)
                    print(f"🔧 Token out of bounds, using UNK token")
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ------------------------------
# Persistent training data storage
# ------------------------------
class PersistentReplayText:
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "training_data.bin")
        self.index_file = os.path.join(data_dir, "data_index.txt")
        self.memory_cache = bytearray()  # Keep recent data in memory
        self.cache_size = 500_000  # 500KB cache
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data size
        self.total_size = 0
        if os.path.exists(self.data_file):
            self.total_size = os.path.getsize(self.data_file)
            # Load recent data into cache
            self._load_recent_to_cache()
    
    def _load_recent_to_cache(self):
        """Load the most recent data into memory cache"""
        if self.total_size == 0:
            return
        
        with open(self.data_file, 'rb') as f:
            # Read from end of file up to cache_size
            start_pos = max(0, self.total_size - self.cache_size)
            f.seek(start_pos)
            self.memory_cache = bytearray(f.read())
    
    def add(self, s: str):
        """Add new text to persistent storage and memory cache"""
        b = s.encode('utf-8', errors='ignore')
        data_with_sep = b + b'\n'
        
        # Append to persistent file
        with open(self.data_file, 'ab') as f:
            f.write(data_with_sep)
        
        # Add to memory cache
        self.memory_cache.extend(data_with_sep)
        
        # Trim cache if too large
        if len(self.memory_cache) > self.cache_size:
            self.memory_cache = self.memory_cache[-self.cache_size//2:]
        
        # Update total size
        self.total_size += len(data_with_sep)
        
        # Log progress periodically
        if self.total_size % 100_000 < len(data_with_sep):
            print(f"📊 Training data: {self.total_size/1e6:.1f}MB stored")
    
    def __len__(self):
        return self.total_size
    
    def sample_batch(self, batch_size=16, block_size=256, device='cpu'):
        """Sample training batches from disk and memory cache"""
        if self.total_size < block_size + 1:
            # Bootstrap with random tokens for initial training
            vocab_size = max(tokenizer.vocab_size, 2)  # Ensure at least 2 tokens (PAD, UNK)
            if vocab_size <= 2:
                # Use simple pattern for very small vocab
                data = torch.tensor([tokenizer.unk_id] * (block_size + 1), dtype=torch.long)
            else:
                data = torch.randint(0, vocab_size, (block_size+1,), dtype=torch.long)
            
            ix = torch.randint(0, 1, (batch_size,))
            x = torch.stack([data[:block_size] for _ in range(batch_size)])
            y = torch.stack([data[1:block_size+1] for _ in range(batch_size)])
            return x.to(device), y.to(device)
        
        # Mixed sampling: 20% from cache (recent), 80% from disk (full history)
        samples_x = []
        samples_y = []
        
        for _ in range(batch_size):
            # 20% chance to sample from cache (recent data), 80% from full history
            if random.random() < 0.2 and len(self.memory_cache) > 100:
                # Sample from memory cache (recent data) - tokenize on the fly
                try:
                    text = self.memory_cache.decode('utf-8', errors='ignore')
                    lines = text.strip().split('\n')
                    if lines:
                        line = random.choice(lines)
                        tokens = tokenizer.encode(line)
                        if len(tokens) > block_size:
                            start_idx = random.randint(0, len(tokens) - block_size - 1)
                            x = tokens[start_idx:start_idx + block_size]
                            y = tokens[start_idx + 1:start_idx + block_size + 1]
                        else:
                            # Pad if too short
                            pad_len = block_size + 1 - len(tokens)
                            padded = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
                            x = padded[:block_size]
                            y = padded[1:block_size + 1]
                    else:
                        x, y = self._sample_from_disk(block_size)
                except:
                    x, y = self._sample_from_disk(block_size)
            else:
                # Sample from disk (full history)
                x, y = self._sample_from_disk(block_size)
            
            samples_x.append(x)
            samples_y.append(y)
        
        x_batch = torch.stack(samples_x)
        y_batch = torch.stack(samples_y)
        return x_batch.to(device), y_batch.to(device)
    
    def _sample_from_disk(self, block_size):
        """Sample a single sequence from disk storage"""
        if self.total_size < 100:
            # Fallback to random tokens (avoid special tokens)
            vocab_size = max(tokenizer.vocab_size, 2)
            if vocab_size <= 2:
                # With very small vocab, create a simple pattern using available tokens
                data = torch.tensor([1] * (block_size + 1), dtype=torch.long)  # Use UNK as last resort
            else:
                # Generate random tokens from non-special token range (start from index 2)
                data = torch.randint(2, vocab_size, (block_size + 1,), dtype=torch.long)
            return data[:block_size], data[1:block_size + 1]
        
        try:
            # Read a random chunk from the file and tokenize
            chunk_size = min(2000, self.total_size)  # Read up to 2KB
            start_pos = random.randint(0, max(0, self.total_size - chunk_size))
            
            with open(self.data_file, 'rb') as f:
                f.seek(start_pos)
                data_bytes = f.read(chunk_size)
            
            # Decode and get random line
            text = data_bytes.decode('utf-8', errors='ignore')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                # Fallback (avoid special tokens)
                vocab_size = max(tokenizer.vocab_size, 2)
                if vocab_size <= 2:
                    data = torch.tensor([1] * (block_size + 1), dtype=torch.long)  # UNK as last resort
                else:
                    data = torch.randint(2, vocab_size, (block_size + 1,), dtype=torch.long)  # Non-special tokens only
                return data[:block_size], data[1:block_size + 1]
            
            # Choose random line and tokenize
            line = random.choice(lines)
            tokens = tokenizer.encode(line)
            
            if len(tokens) > block_size:
                start_idx = random.randint(0, len(tokens) - block_size - 1)
                x = tokens[start_idx:start_idx + block_size]
                y = tokens[start_idx + 1:start_idx + block_size + 1]
            else:
                # Pad if too short
                pad_len = block_size + 1 - len(tokens)
                padded = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
                x = padded[:block_size]
                y = padded[1:block_size + 1]
            
            return x, y
            
        except Exception:
            # Fallback to random tokens
            vocab_size = max(tokenizer.vocab_size, 2)
            if vocab_size <= 2:
                data = torch.tensor([tokenizer.unk_id] * (block_size + 1), dtype=torch.long)
            else:
                data = torch.randint(0, vocab_size, (block_size + 1,), dtype=torch.long)
            return data[:block_size], data[1:block_size + 1]
    
    def get_stats(self):
        """Get storage statistics"""
        return {
            'total_size_mb': self.total_size / 1e6,
            'cache_size_kb': len(self.memory_cache) / 1024,
            'data_file': self.data_file
        }

# ------------------------------
# EMA (Exponential Moving Average) for model weights
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    # Check if parameter size has changed (vocabulary expansion)
                    if param.data.shape != self.shadow[name].shape:
                        # Expand shadow weights for vocabulary expansion
                        old_size = self.shadow[name].shape[0]
                        new_size = param.data.shape[0]
                        
                        if len(param.data.shape) == 2 and new_size > old_size:  # Embedding or Linear layer
                            # Create new shadow tensor with expanded size
                            new_shadow = torch.zeros_like(param.data)
                            new_shadow[:old_size] = self.shadow[name]
                            new_shadow[old_size:] = param.data[old_size:].clone()  # Initialize new weights
                            self.shadow[name] = new_shadow
                        else:
                            # For other shape mismatches, reinitialize
                            self.shadow[name] = param.data.clone()
                    
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                else:
                    # New parameter, initialize shadow
                    self.shadow[name] = param.data.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                
                # Handle size mismatch (vocabulary expansion)
                if param.data.shape != self.shadow[name].shape:
                    # Update shadow to match current parameter size
                    self.update(model)
                
                if param.data.shape == self.shadow[name].shape:
                    param.data = self.shadow[name]
                else:
                    # Skip if still mismatched
                    print(f"⚠️  Skipping EMA shadow for {name} due to size mismatch")
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                # Handle size mismatch
                if param.data.shape == self.original[name].shape:
                    param.data = self.original[name]
                else:
                    print(f"⚠️  Cannot restore {name} due to size mismatch")
        self.original.clear()

# ------------------------------
# Cosine Learning Rate Scheduler
# ------------------------------
class CosineScheduler:
    def __init__(self, optimizer, warmup_steps=100, max_steps=10000, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# ------------------------------
# Checkpoint save/load functions
# ------------------------------
def save_checkpoint(model, optimizer, replay, filename="model_checkpoint.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'block_size': block_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'dropout': dropout
        },
        'data_stats': replay.get_stats()
    }
    torch.save(checkpoint, filename)
    print(f"💾 Model saved to {filename}")

def load_checkpoint(model, optimizer, replay, filename="model_checkpoint.pt"):
    if not os.path.exists(filename):
        print(f"❌ Checkpoint {filename} not found")
        return False
    
    try:
        checkpoint = torch.load(filename, map_location=device)
        
        # Check if model configuration matches
        if 'config' in checkpoint:
            config = checkpoint['config']
            if (config['block_size'] != block_size or 
                config['n_embd'] != n_embd or 
                config['n_head'] != n_head or 
                config['n_layer'] != n_layer):
                print(f"⚠️  Model configuration mismatch!")
                print(f"   Checkpoint: block_size={config['block_size']}, n_embd={config['n_embd']}, n_head={config['n_head']}, n_layer={config['n_layer']}")
                print(f"   Current:    block_size={block_size}, n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
                print("   Cannot load incompatible checkpoint.")
                return False
            
            # Handle dynamic vocabulary size - expand model if needed
            checkpoint_vocab_size = config.get('vocab_size', 0)
            if checkpoint_vocab_size > model.current_vocab_size:
                print(f"🔄 Expanding model vocabulary from {model.current_vocab_size} to {checkpoint_vocab_size}")
                model.expand_vocab(checkpoint_vocab_size)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Note: Replay buffer is now persistent on disk, no need to save/load
        
        print(f"✅ Model loaded from {filename}")
        stats = replay.get_stats()
        print(f"📚 Training data: {stats['total_size_mb']:.1f}MB on disk, {stats['cache_size_kb']:.1f}KB in cache")
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

# ------------------------------
# Setup
# ------------------------------
# Initialize with dynamic vocabulary (starting from 2 for PAD and UNK tokens)
initial_vocab_size = max(tokenizer.vocab_size, 2)
model = TinyGPT(initial_vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
replay = PersistentReplayText()  # Unlimited persistent storage

# Enhanced training components
if use_ema:
    ema = EMA(model, decay=ema_decay)
else:
    ema = None

scheduler = CosineScheduler(opt, warmup_steps=100, max_steps=10000)

# Try to load existing checkpoint
if load_checkpoint(model, opt, replay):
    pass  # Successfully loaded
else:
    # seed with a tiny primer so sampling isn't entirely impossible at start
    replay.add(" ")

print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print("Tiny LLM demo started. Type text to feed the model. Commands: /gen, /temp=, /steps=, /save, /load, /stats, /quit\n")

temperature = 1.1
train_steps_per_pulse = 20   
batch_size = 4


def training_pulse(steps=50):
    model.train()
    avg_loss = 0.0
    for i in range(steps):
        x, y = replay.sample_batch(batch_size=batch_size, block_size=block_size, device=device)
        logits, loss = model(x, y)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # Update learning rate schedule
        lr = scheduler.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
            
        avg_loss += loss.item()
    
    return avg_loss / max(1, steps)

def process_text_line(text: str, show_generation=True):
    """Process a single line of text just like normal user input"""
    # Add new text to dynamic SentencePiece tokenizer
    tokenizer.add_text(text)
    
    # Expand model vocabulary if needed
    if tokenizer.vocab_size > model.current_vocab_size:
        model.expand_vocab(tokenizer.vocab_size)
    
    # Add to training data
    replay.add(text)
    
    # Train on this text
    t0 = time.time()
    avg_loss = training_pulse(train_steps_per_pulse)
    dt = (time.time()-t0)*1000
    print(f"⚙️  Trained {train_steps_per_pulse} steps | avg loss {avg_loss:.3f} | {dt:.0f} ms")

    # Generate sample if requested
    if show_generation:
        try:
            # print("\n🧪 Sampling after update...")
            sample = generate_sample(prefix="", max_new_tokens=100, adaptive_length=True)
            print(f"\n--- Generation (temp={temperature}) ---\n{sample}\n--- end ---")
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            print("   Training completed successfully, but generation had issues.")

def generate_sample(prefix: str = "", max_new_tokens=200, adaptive_length=True):
    # Use EMA weights for generation if available
    if ema is not None:
        ema.apply_shadow(model)
    
    try:
        with torch.no_grad():
            if prefix:
                start = tokenizer.encode(prefix).unsqueeze(0).to(device)
            else:
                # Start with empty context - no initial token, let model generate from scratch
                start = torch.empty((1, 0), dtype=torch.long, device=device)
            
            # Use dynamic top-k based on vocabulary size
            effective_k = min(100, max(10, tokenizer.vocab_size // 2))
            out = model.generate(start, max_new_tokens=max_new_tokens, 
                               temperature=temperature, top_k=effective_k, top_p=top_p,
                               adaptive_length=adaptive_length)
            gen = out[0].tolist()
            
            if prefix:
                gen = gen[len(start[0]):]  # drop the prefix tokens
            # For empty start, all generated tokens are new
            
            return tokenizer.decode(torch.tensor(gen))
    finally:
        # Restore original weights
        if ema is not None:
            ema.restore(model)

# ------------------------------
# REPL
# ------------------------------
help_tip = "Enhanced TinyGPT with Dynamic Vocabulary, RoPE+NTK, weight sharing, cosine LR, EMA, and top-p sampling\nCommands: /gen [prefix], /temp=1.2, /topp=0.9, /steps=200, /context=1024, /load_txt [txt_file], /train [txt_file], /save [filename], /load [filename], /stats, /reset, /quit\nNTK Scaling: Supports context lengths up to 2048 tokens"
print(help_tip)

while True:
    try:
        user = input("\n✍️  Input (or command): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n💾 Auto-saving before exit...")
        save_checkpoint(model, opt, replay)
        print("Bye.")
        break

    if not user:
        continue

    if user.startswith("/reset"):
        # Reset model to initial state
        if input("⚠️  This will delete all training data and reset the model. Continue? (y/n): ").lower().strip() == 'y':
            try:
                # Remove model files
                if os.path.exists("dynamic_vocab.json"):
                    os.remove("dynamic_vocab.json")
                if os.path.exists("model_checkpoint.pt"):
                    os.remove("model_checkpoint.pt")
                if os.path.exists("training_data"):
                    import shutil
                    shutil.rmtree("training_data")
                
                # Remove SentencePiece files
                if os.path.exists("tokenizer.model"):
                    os.remove("tokenizer.model")
                if os.path.exists("tokenizer.vocab"):
                    os.remove("tokenizer.vocab")
                if os.path.exists("sp_training_data.txt"):
                    os.remove("sp_training_data.txt")
                if os.path.exists("sp_stats.json"):
                    os.remove("sp_stats.json")
                
                print("🔄 Model and tokenizer reset complete. Please restart the program for changes to take effect.")
                break
            except Exception as e:
                print(f"❌ Reset failed: {e}")
        else:
            print("Reset cancelled.")
        continue

    if user.startswith("/quit"):
        # Auto-save before quitting
        save_checkpoint(model, opt, replay)
        print("Bye.")
        break

    if user.startswith("/temp="):
        try:
            temperature = float(user.split("=",1)[1])
            print(f"Set temperature = {temperature}")
        except:
            print("Usage: /temp=1.2")
        continue

    if user.startswith("/topp="):
        try:
            top_p = float(user.split("=",1)[1])
            print(f"Set top_p = {top_p}")
        except:
            print("Usage: /topp=0.9")
        continue

    if user.startswith("/steps="):
        try:
            train_steps_per_pulse = int(user.split("=",1)[1])
            print(f"Set train_steps_per_pulse = {train_steps_per_pulse}")
        except:
            print("Usage: /steps=100")
        continue

    if user.startswith("/context="):
        try:
            new_block_size = int(user.split("=",1)[1])
            if new_block_size < 64:
                print("❌ Context size must be at least 64")
                continue
            if new_block_size > max_context:
                print(f"❌ Context size cannot exceed {max_context} (NTK scaling limit)")
                continue
            
            # Update global block_size
            globals()['block_size'] = new_block_size
            print(f"✅ Set context length = {new_block_size}")
            print("🔧 NTK scaling will automatically handle sequences longer than the original training length")
        except:
            print(f"Usage: /context=1024 (max: {max_context})")
        continue

    if user.startswith("/gen"):
        # /gen optional_prefix
        prefix = user[4:].lstrip()
        print("\n🧪 Sampling...")
        # Use adaptive length for better natural responses
        adaptive = len(prefix.strip()) > 0  # Use adaptive for prefixed generation
        max_tokens = 300 if adaptive else 500  # Shorter max when adaptive
        txt = generate_sample(prefix=prefix, max_new_tokens=max_tokens, adaptive_length=adaptive)
        print(f"\n--- Generation (temp={temperature}) ---\n{txt}\n--- end ---")
        continue

    if user.startswith("/save"):
        # /save optional_filename
        parts = user.split(None, 1)
        filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
        save_checkpoint(model, opt, replay, filename)
        continue

    if user.startswith("/load_txt"):
        # /load_txt filename.txt
        parts = user.split(None, 1)
        if len(parts) < 2:
            print("Usage: /load_txt filename.txt")
            continue
        
        filename = parts[1]
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
        
        try:
            print(f"📖 Loading and processing file: {filename}")
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            if total_lines == 0:
                print("❌ File is empty")
                continue
            
            print(f"📚 Found {total_lines} lines, processing line by line...\n")
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    # print(f"⏭️  Skipping empty line {i}/{total_lines}")
                    continue
                
                print(f"\n📝 Processing line {i}/{total_lines}: {line}")
                
                # Process this line like normal user input
                process_text_line(line, show_generation=True)
                
                # Add a small delay to make output readable
                import time
                time.sleep(0.1)
            
            print(f"\n✅ Finished processing {filename}")
            
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
        continue

    if user.startswith("/load"):
        # /load optional_filename (for model checkpoints)
        parts = user.split(None, 1)
        filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
        load_checkpoint(model, opt, replay, filename)
        continue


    if user.startswith("/train"):
        # /train filename.txt - Legacy command, now same as /load_txt
        parts = user.split(None, 1)
        if len(parts) < 2:
            print("Usage: /train filename.txt (note: consider using /load_txt instead)")
            continue
        
        filename = parts[1]
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
        
        try:
            print(f"📖 Training on file: {filename}")
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            if total_lines == 0:
                print("❌ File is empty")
                continue
            
            print(f"📚 Found {total_lines} lines, training line by line...\n")
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    # print(f"⏭️  Skipping empty line {i}/{total_lines}")
                    continue
                
                print(f"\n📝 Training on line {i}/{total_lines}: {line}")
                
                # Process this line like normal user input
                process_text_line(line, show_generation=True)
            
            print(f"\n✅ Finished training on {filename}")
            
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
        continue

    if user.startswith("/stats"):
        stats = replay.get_stats()
        print(f"📊 Enhanced TinyGPT Statistics:")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"   Vocab size: {tokenizer.vocab_size} (Dynamic)")
        print(f"   Context length: {block_size} (max: {max_context} with NTK)")
        print(f"   RoPE+NTK: {'✅' if use_rope else '❌'}")
        print(f"   Weight sharing: {'✅' if tie_weights else '❌'}")
        print(f"   EMA: {'✅' if use_ema else '❌'}")
        print(f"   Current LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
        print(f"   Training steps: {scheduler.step_count}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
        # Add vocabulary statistics
        vocab_stats = tokenizer.get_stats()
        print(f"🔤 Vocabulary:")
        print(f"   Vocab size: {vocab_stats['vocab_size']}")
        print(f"   Model type: {vocab_stats['model_type']}")
        print(f"   Model loaded: {vocab_stats['model_loaded']}")
        print(f"   Model path: {vocab_stats['model_path']}")
        print(f"   Total texts seen: {vocab_stats['total_texts_seen']}")
        print(f"📚 Training Data:")
        print(f"   Total data stored: {stats['total_size_mb']:.1f} MB")
        print(f"   Memory cache: {stats['cache_size_kb']:.1f} KB")
        print(f"   Data file: {stats['data_file']}")
        continue

    # Regular text: process like normal user input
    process_text_line(user, show_generation=True)


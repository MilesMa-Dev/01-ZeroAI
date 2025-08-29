# tiny_llm_demo.py
# Minimal byte-level Transformer LM with online training "pulses".
# Author: Miles + GPT-5 Thinking
# Run: python tiny_llm_demo.py

import math, sys, os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Config (updated for larger context)
# ------------------------------
vocab_size   = 256           # byte-level
block_size   = 256          # context length
n_embd       = 256           # model width (increased for larger context)
n_head       = 8             # more attention heads
n_layer      = 4             # deeper model
dropout      = 0.1           # small dropout for regularization
device       = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 1337
torch.manual_seed(seed)
random.seed(seed)

# ------------------------------
# Utilities: byte tokenizer
# ------------------------------
def encode_bytes(s: str) -> torch.Tensor:
    # UTF-8 to bytes 0..255
    b = s.encode('utf-8', errors='ignore')
    return torch.tensor(list(b), dtype=torch.long)

def decode_bytes(t: torch.Tensor) -> str:
    by = bytes([int(x) for x in t.tolist()])
    try:
        return by.decode('utf-8', errors='ignore')
    except:
        return by.decode('latin1', errors='ignore')

# ------------------------------
# Model: tiny GPT-like Transformer
# ------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key    = nn.Linear(n_embd, n_embd, bias=False)
        self.query  = nn.Linear(n_embd, n_embd, bias=False)
        self.value  = nn.Linear(n_embd, n_embd, bias=False)
        self.proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # causal mask
        mask = torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)
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
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)                         # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                           # (B,T,vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
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
            # Bootstrap with random bytes for initial training
            data = torch.randint(0, 256, (block_size+1,), dtype=torch.long)
            ix = torch.randint(0, 1, (batch_size,))
            x = torch.stack([data[:block_size] for _ in range(batch_size)])
            y = torch.stack([data[1:block_size+1] for _ in range(batch_size)])
            return x.to(device), y.to(device)
        
        # Mixed sampling: 20% from cache (recent), 80% from disk (full history)
        samples_x = []
        samples_y = []
        
        for _ in range(batch_size):
            # 20% chance to sample from cache (recent data), 80% from full history
            if random.random() < 0.2 and len(self.memory_cache) > block_size + 1:
                # Sample from memory cache (recent data)
                data = torch.tensor(list(self.memory_cache), dtype=torch.long)
                max_start = data.numel() - block_size - 1
                start_idx = torch.randint(0, max(1, max_start), (1,)).item()
                x = data[start_idx:start_idx + block_size]
                y = data[start_idx + 1:start_idx + block_size + 1]
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
        if self.total_size < block_size + 1:
            # Fallback to random data
            data = torch.randint(0, 256, (block_size + 1,), dtype=torch.long)
            return data[:block_size], data[1:block_size + 1]
        
        # Choose random position in file
        max_start = self.total_size - block_size - 1
        start_pos = random.randint(0, max_start)
        
        # Read data from file
        with open(self.data_file, 'rb') as f:
            f.seek(start_pos)
            data_bytes = f.read(block_size + 1)
        
        # Convert to tensor
        if len(data_bytes) < block_size + 1:
            # Handle edge case - pad with random bytes
            data_bytes += bytes([random.randint(0, 255) for _ in range(block_size + 1 - len(data_bytes))])
        
        data = torch.tensor(list(data_bytes), dtype=torch.long)
        return data[:block_size], data[1:block_size + 1]
    
    def get_stats(self):
        """Get storage statistics"""
        return {
            'total_size_mb': self.total_size / 1e6,
            'cache_size_kb': len(self.memory_cache) / 1024,
            'data_file': self.data_file
        }

# ------------------------------
# Checkpoint save/load functions
# ------------------------------
def save_checkpoint(model, optimizer, replay, filename="model_checkpoint.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': vocab_size,
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
model = TinyGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
replay = PersistentReplayText()  # Unlimited persistent storage

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
    avg = 0.0
    for i in range(steps):
        x, y = replay.sample_batch(batch_size=batch_size, block_size=block_size, device=device)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        avg += loss.item()
    return avg / max(1, steps)

def generate_sample(prefix:str="", max_new_tokens=200):
    with torch.no_grad():
        if prefix:
            start = encode_bytes(prefix).unsqueeze(0).to(device)
        else:
            # start with a single space byte
            start = torch.tensor([[32]], dtype=torch.long, device=device)
        out = model.generate(start, max_new_tokens=max_new_tokens, temperature=temperature, top_k=100)
        gen = out[0].tolist()
        if prefix:
            gen = gen[len(start[0]):]  # drop the prefix bytes
        return decode_bytes(torch.tensor(gen))

# ------------------------------
# REPL
# ------------------------------
help_tip = "Enter text to learn from, or commands: /gen [prefix], /temp=1.2, /steps=200, /train [txt_file], /save [filename], /load [filename], /stats, /quit"
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

    if user.startswith("/steps="):
        try:
            train_steps_per_pulse = int(user.split("=",1)[1])
            print(f"Set train_steps_per_pulse = {train_steps_per_pulse}")
        except:
            print("Usage: /steps=100")
        continue

    if user.startswith("/gen"):
        # /gen optional_prefix
        prefix = user[4:].lstrip()
        print("\n🧪 Sampling...")
        txt = generate_sample(prefix=prefix, max_new_tokens=500)
        print(f"\n--- Generation (temp={temperature}) ---\n{txt}\n--- end ---")
        continue

    if user.startswith("/save"):
        # /save optional_filename
        parts = user.split(None, 1)
        filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
        save_checkpoint(model, opt, replay, filename)
        continue

    if user.startswith("/load"):
        # /load optional_filename
        parts = user.split(None, 1)
        filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
        load_checkpoint(model, opt, replay, filename)
        continue


    if user.startswith("/train"):
        # /train filename.txt
        parts = user.split(None, 1)
        if len(parts) < 2:
            print("Usage: /train filename.txt")
            continue
        
        filename = parts[1]
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
        
        try:
            print(f"📖 Reading file: {filename}")
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            if total_lines == 0:
                print("❌ File is empty")
                continue
            
            print(f"📚 Found {total_lines} lines, training line by line...")
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                print(f"\n📝 Training on line {i}/{total_lines}: {line}")
                
                # Add line to training data
                replay.add(line)
                
                # Train on this line
                t0 = time.time()
                avg_loss = training_pulse(train_steps_per_pulse)
                dt = (time.time()-t0)*1000
                print(f"⚙️  Trained {train_steps_per_pulse} steps | avg loss {avg_loss:.3f} | {dt:.0f} ms")
                
                # Generate sample after each line training (like manual input)
                print("🧪 Sampling after update...")
                sample = generate_sample(prefix="", max_new_tokens=300)
                print(f"\n--- Generation (temp={temperature}) ---\n{sample}\n--- end ---")
            
            print(f"\n✅ Finished training on {filename}")
            
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
        continue

    if user.startswith("/stats"):
        stats = replay.get_stats()
        print(f"📊 Training Data Statistics:")
        print(f"   Total data stored: {stats['total_size_mb']:.1f} MB")
        print(f"   Memory cache: {stats['cache_size_kb']:.1f} KB")
        print(f"   Data file: {stats['data_file']}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        continue

    # Regular text: add to replay, train a small pulse, then show generation
    replay.add(user)
    t0 = time.time()
    avg_loss = training_pulse(train_steps_per_pulse)
    dt = (time.time()-t0)*1000
    print(f"⚙️  Trained {train_steps_per_pulse} steps | avg loss {avg_loss:.3f} | {dt:.0f} ms")

    print("\n🧪 Sampling after update...")
    sample = generate_sample(prefix="", max_new_tokens=300)
    print(f"\n--- Generation (temp={temperature}) ---\n{sample}\n--- end ---")


#!/usr/bin/env python3
"""
Dynamic SentencePiece Unigram Tokenizer for TinyGPT
"""
import sentencepiece as spm
import os
import torch
import json
from typing import List, Union

class DynamicSPTokenizer:
    """Dynamic SentencePiece Unigram tokenizer that grows incrementally"""
    
    def __init__(self, model_path: str = "tokenizer.model", base_vocab_size: int = 500):
        self.model_path = model_path
        self.base_vocab_size = base_vocab_size
        self.current_vocab_size = base_vocab_size
        self.sp = None
        self.training_data = []  # Accumulate training data
        self.data_file = "sp_training_data.txt"
        self.stats_file = "sp_stats.json"
        
        # Load existing stats
        self.stats = self._load_stats()
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"🔧 Dynamic SentencePiece starting from scratch")
            print(f"🔧 Will train with vocab_size={base_vocab_size} on first input")
            
            # Bootstrap with existing large corpus if available
            self._bootstrap_with_large_corpus()
    
    def _bootstrap_with_large_corpus(self):
        """Skip bootstrap - wait for actual user input"""
        print("🔧 No initial training - waiting for user input to build vocabulary from scratch")
    
    def _load_stats(self):
        """Load tokenizer statistics"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'total_texts_seen': 0,
            'current_vocab_size': self.base_vocab_size,
            'expansion_history': []
        }
    
    def _save_stats(self):
        """Save tokenizer statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save stats: {e}")
    
    def add_text(self, text: str):
        """Add new text and potentially expand vocabulary"""
        # Add to training data accumulator
        self.training_data.append(text)
        self.stats['total_texts_seen'] += 1
        
        # Save to persistent training data file
        with open(self.data_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        
        # Decide if we need to expand vocabulary
        should_expand = self._should_expand_vocab()
        
        if should_expand:
            self._expand_vocabulary()
        elif self.sp is None:
            # First time training
            print(f"🔧 Training initial SentencePiece model (vocab_size={self.current_vocab_size})")
            self._retrain_model()
    
    def _should_expand_vocab(self) -> bool:
        """Determine if vocabulary should be expanded"""
        # Expand every 50 new texts, with exponential backoff
        texts_seen = self.stats['total_texts_seen']
        expansions = len(self.stats['expansion_history'])
        
        # Exponential growth pattern: expand at 50, 150, 350, 750, etc.
        next_expansion_point = 50 * (2 ** expansions)
        
        return texts_seen >= next_expansion_point and self.current_vocab_size < 8192
    
    def _expand_vocabulary(self):
        """Expand vocabulary and retrain model"""
        old_size = self.current_vocab_size
        # Increase vocab size by 50% each time, capped at 8192
        self.current_vocab_size = min(8192, int(self.current_vocab_size * 1.5))
        
        print(f"📝 Expanding vocabulary from {old_size} to {self.current_vocab_size} (seen {self.stats['total_texts_seen']} texts)")
        
        # Record expansion
        self.stats['expansion_history'].append({
            'from_size': old_size,
            'to_size': self.current_vocab_size,
            'at_text_count': self.stats['total_texts_seen']
        })
        self.stats['current_vocab_size'] = self.current_vocab_size
        
        self._retrain_model()
    
    def _retrain_model(self):
        """Retrain SentencePiece model with current data"""
        if not os.path.exists(self.data_file):
            print("⚠️ No training data available")
            return
        
        print(f"🔧 Retraining SentencePiece model with vocab_size={self.current_vocab_size}...")
        
        # Calculate character coverage based on vocab size
        char_coverage = 0.9995 if self.current_vocab_size >= 4000 else 0.98
        
        try:
            # Train SentencePiece model
            spm.SentencePieceTrainer.train(
                input=self.data_file,
                model_prefix='tokenizer',
                vocab_size=self.current_vocab_size,
                model_type='unigram',  # Use Unigram model
                character_coverage=char_coverage,
                split_by_unicode_script=False,  # Don't split by script (for mixed languages)
                byte_fallback=True,  # Handle unknown characters
                normalization_rule_name='identity',  # No normalization
                train_extremely_large_corpus=False,
                user_defined_symbols=['<BOS>', '<EOS>'],  # Only custom special tokens
                control_symbols=['<PAD>'],  # Control symbols
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # Load the trained model
            self.load_model()
            self._save_stats()
            print(f"✅ SentencePiece model retrained! Vocabulary size: {self.sp.get_piece_size()}")
            
        except Exception as e:
            print(f"⚠️ Failed to retrain model: {e}")
            if self.sp is None:
                # Extract required vocab size from error message if available
                error_str = str(e)
                if "Please set it to a value <=" in error_str:
                    try:
                        max_vocab = int(error_str.split("<=")[1].split(".")[0].strip())
                        if max_vocab < self.current_vocab_size:
                            print(f"🔧 Adjusting vocab size to maximum allowed: {max_vocab}")
                            self.current_vocab_size = max_vocab
                            self.stats['current_vocab_size'] = self.current_vocab_size
                            self._retrain_model()
                            return
                    except:
                        pass
                
                # Fallback: try with much smaller vocab
                if self.current_vocab_size > 500:
                    print("🔧 Trying with smaller vocabulary...")
                    self.current_vocab_size = min(500, max(300, self.current_vocab_size // 3))
                    self.stats['current_vocab_size'] = self.current_vocab_size
                    self._retrain_model()
    
    def load_model(self):
        """Load existing SentencePiece model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)
        
        # Don't reset current_vocab_size - keep the target size for expansion logic
        actual_vocab_size = self.sp.get_piece_size()
        if hasattr(self, 'current_vocab_size') and self.current_vocab_size != actual_vocab_size:
            print(f"📊 Note: Target vocab size: {self.current_vocab_size}, Actual vocab size: {actual_vocab_size}")
        
        print(f"✅ Loaded SentencePiece model: {self.model_path}")
        print(f"📊 Vocabulary size: {actual_vocab_size}")
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        if self.sp is None:
            # Auto-train on first use - this will be triggered by add_text
            print(f"⚠️ No model available - call add_text() first")
            return torch.tensor([1], dtype=torch.long)  # Return UNK token
        
        # Encode text
        ids = self.sp.encode_as_ids(text)
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """Decode token IDs to text"""
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Decode IDs to text
        return self.sp.decode_ids(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp is None:
            return self.current_vocab_size
        return self.sp.get_piece_size()
    
    @property
    def vocab_size(self) -> int:
        """Compatibility property for vocab_size"""
        return self.get_vocab_size()
    
    def get_piece(self, id: int) -> str:
        """Get piece by ID"""
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        return self.sp.id_to_piece(id)
    
    def get_piece_id(self, piece: str) -> int:
        """Get ID by piece"""
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        return self.sp.piece_to_id(piece)
    
    @property
    def pad_id(self) -> int:
        return 0
    
    @property
    def unk_id(self) -> int:
        return 1
    
    @property
    def bos_id(self) -> int:
        return 2
    
    @property
    def eos_id(self) -> int:
        return 3
    
    def get_stats(self):
        """Get tokenizer statistics"""
        base_stats = {
            'vocab_size': self.current_vocab_size,
            'model_loaded': self.sp is not None,
            'model_path': self.model_path,
            'model_type': 'Dynamic SentencePiece Unigram',
            'special_tokens': {
                'PAD': self.pad_id,
                'UNK': self.unk_id, 
                'BOS': self.bos_id,
                'EOS': self.eos_id
            },
            'total_texts_seen': self.stats['total_texts_seen'],
            'expansion_history': self.stats['expansion_history']
        }
        
        if self.sp is not None:
            base_stats['actual_vocab_size'] = self.sp.get_piece_size()
        
        return base_stats


if __name__ == "__main__":
    # Test the dynamic tokenizer
    tokenizer = DynamicSPTokenizer(base_vocab_size=500)
    
    # Test with mixed text
    test_texts = [
        "Hello world!",
        "你好世界！",
        "This is a test.",
        "这是一个测试。",
        "Machine learning is amazing!",
        "人工智能很棒！"
    ]
    
    print("=== Testing Dynamic SentencePiece Tokenizer ===")
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Adding text {i+1}: {text} ---")
        
        # Add text (may trigger expansion)
        tokenizer.add_text(text)
        
        # Encode and decode
        if tokenizer.sp is not None:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            print(f"Encoded IDs: {ids}")
            print(f"Decoded: {decoded}")
        
        # Show stats
        stats = tokenizer.get_stats()
        print(f"Vocab size: {stats['vocab_size']} | Texts seen: {stats['total_texts_seen']}")
    
    print(f"\n=== Final Stats ===")
    final_stats = tokenizer.get_stats()
    for key, value in final_stats.items():
        print(f"{key}: {value}")
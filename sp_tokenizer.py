#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic SentencePiece Unigram Tokenizer for TinyGPT (robust version)
- Fixes the "Vocabulary size is smaller than required_chars" error
- Removes duplicated special symbol declarations
- Adds hard_vocab_limit=False and robust retries
- Dynamically adjusts vocab_size / character_coverage when needed
"""

import os
import re
import json
import torch
import sentencepiece as spm
from typing import List, Union, Dict, Any


class DynamicSPTokenizer:
    """
    Dynamic SentencePiece Unigram tokenizer that grows incrementally.

    Key behaviors:
    - Starts with a base vocab size; collects texts and (re)trains SPM.
    - Expands vocab in small steps (e.g., +20%) as more texts are seen.
    - Uses robust training with:
        * hard_vocab_limit=False
        * dynamic character_coverage based on vocab size
        * fallback to increase vocab_size or lower character_coverage
          when "required_chars > vocab_size" occurs.
    - Avoids duplicating special tokens (only meta ids are set).
    """

    def __init__(self,
                 model_path: str = "tokenizer.model",
                 base_vocab_size: int = 1000,
                 data_file: str = "sp_training_data.txt",
                 stats_file: str = "sp_stats.json"):
        self.model_path = model_path
        self.base_vocab_size = int(base_vocab_size)
        self.current_vocab_size = int(base_vocab_size)
        self.sp: spm.SentencePieceProcessor | None = None

        # persistent data files
        self.data_file = data_file
        self.stats_file = stats_file

        # in-memory accumulator (optional / not strictly needed)
        self.training_data: List[str] = []

        # load running stats
        self.stats: Dict[str, Any] = self._load_stats()

        # try to load existing model or bootstrap
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("🔧 Dynamic SentencePiece starting from scratch")
            print(f"🔧 Will train with vocab_size={self.current_vocab_size} on first input")
            self._bootstrap_with_large_corpus()

    # -----------------------------
    # Utilities: stats & bootstrap
    # -----------------------------
    def _load_stats(self) -> Dict[str, Any]:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    # ensure keys exist
                    obj.setdefault("total_texts_seen", 0)
                    obj.setdefault("current_vocab_size", self.base_vocab_size)
                    obj.setdefault("expansion_history", [])
                    return obj
            except Exception:
                pass
        return {
            "total_texts_seen": 0,
            "current_vocab_size": self.base_vocab_size,
            "expansion_history": []
        }

    def _save_stats(self) -> None:
        self.stats["current_vocab_size"] = self.current_vocab_size
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save stats: {e}")

    def _bootstrap_with_large_corpus(self) -> None:
        """
        If there is existing training data, immediately train a first model.
        """
        if os.path.exists(self.data_file):
            print(f"🔧 Found existing training data: {self.data_file}")
            with open(self.data_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.rstrip("\n") for ln in f]
            if lines:
                self.stats["total_texts_seen"] = len(lines)
                print(f"🔧 Bootstrapping tokenizer with {len(lines)} texts...")
                self._retrain_model()
            else:
                print("🔧 No non-empty lines in training data; waiting for user input")
        else:
            print("🔧 No initial training data; waiting for first input")

    # -----------------------------
    # Public API
    # -----------------------------
    def add_text(self, text: str) -> None:
        """
        Add new text and retrain model each time (art experiment mode).
        Expands vocab periodically for gradual growth.
        """
        if not isinstance(text, str):
            return
        text = text.strip("\n")
        if not text:
            return

        # accumulate to memory & disk
        self.training_data.append(text)
        self.stats["total_texts_seen"] += 1
        with open(self.data_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        # decide whether to expand vocab
        if self._should_expand_vocab():
            self._expand_vocabulary()
        else:
            self._retrain_model()

    def encode(self, text: str) -> torch.Tensor:
        if self.sp is None:
            print("⚠️ No model available - call add_text() first to trigger training")
            return torch.tensor([self.unk_id], dtype=torch.long)
        ids = self.sp.encode(text, out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode(ids)

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        actual = self.sp.get_piece_size()
        if self.current_vocab_size != actual:
            print(f"📊 Note: target vocab_size={self.current_vocab_size}, actual={actual}")
        print(f"✅ Loaded SentencePiece model: {self.model_path} (vocab={actual})")

    def get_vocab_size(self) -> int:
        return self.sp.get_piece_size() if self.sp is not None else self.current_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def get_piece(self, piece_id: int) -> str:
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        return self.sp.id_to_piece(piece_id)

    def get_piece_id(self, piece: str) -> int:
        if self.sp is None:
            raise RuntimeError("Model not loaded")
        return self.sp.piece_to_id(piece)

    # Special token ids (meta pieces in SPM)
    @property
    def pad_id(self) -> int: return 0
    @property
    def unk_id(self) -> int: return 1
    @property
    def bos_id(self) -> int: return 2
    @property
    def eos_id(self) -> int: return 3

    def get_stats(self) -> Dict[str, Any]:
        base = {
            "model_loaded": self.sp is not None,
            "model_path": self.model_path,
            "model_type": "Dynamic SentencePiece Unigram",
            "current_vocab_target": self.current_vocab_size,
            "total_texts_seen": self.stats["total_texts_seen"],
            "expansion_history": self.stats["expansion_history"],
            "special_tokens": {"PAD": self.pad_id, "UNK": self.unk_id, "BOS": self.bos_id, "EOS": self.eos_id}
        }
        if self.sp is not None:
            base["actual_vocab_size"] = self.sp.get_piece_size()
        return base

    # -----------------------------
    # Internals: vocab growth
    # -----------------------------
    def _should_expand_vocab(self) -> bool:
        """
        Expand every 5, 10, 15, ... texts; cap at 8192.
        """
        seen = self.stats["total_texts_seen"]
        next_point = 5 * (len(self.stats["expansion_history"]) + 1)
        if seen >= next_point and self.current_vocab_size < 8192:
            return True
        return False

    def _expand_vocabulary(self) -> None:
        old = self.current_vocab_size
        self.current_vocab_size = min(8192, max(old + 1, int(old * 1.2)))  # +20% (at least +1)
        self.stats["expansion_history"].append({
            "from_size": old,
            "to_size": self.current_vocab_size,
            "at_text_count": self.stats["total_texts_seen"]
        })
        # print(f"📝 Expanding vocab {old} → {self.current_vocab_size} at #{self.stats['total_texts_seen']} texts")
        self._retrain_model()

    # -----------------------------
    # Internals: robust (re)training
    # -----------------------------
    def _choose_char_coverage(self, vsize: int) -> float:
        """
        Smaller vocab → slightly lower coverage, but never below 0.98
        (some SPM builds enforce [0.98, 1.0]).
        """
        if vsize >= 4000: return 0.9995
        if vsize >= 2000: return 0.995
        if vsize >= 1000: return 0.99
        # minimum allowed by your SPM build
        return 0.98


    def _train_once(self, vsize: int, coverage: float) -> None:
        """
        Single SPM training call; avoids duplicate special symbol declarations.
        Writes tokenizer.model / tokenizer.vocab.
        """
        spm.SentencePieceTrainer.train(
            input=self.data_file,
            model_prefix=os.path.splitext(self.model_path)[0],
            vocab_size=vsize,
            model_type="unigram",
            character_coverage=coverage,
            split_by_unicode_script=False,
            byte_fallback=True,
            normalization_rule_name="identity",
            train_extremely_large_corpus=False,
            # critical: do not hard-fail if required_chars slightly exceed
            hard_vocab_limit=False,
            # meta pieces only; do NOT also pass user_defined/control symbols of same meaning
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id
        )

    def _silence_cpp_logs(self):
        """
        Context manager to silence SentencePiece C++ stderr logs.
        """
        class _Silencer:
            def __enter__(self_inner):
                import sys
                self_inner.stderr_fd = sys.stderr.fileno()
                self_inner.devnull = open(os.devnull, "w")
                self_inner.devnull_fd = self_inner.devnull.fileno()
                self_inner.backup = os.dup(self_inner.stderr_fd)
                os.dup2(self_inner.devnull_fd, self_inner.stderr_fd)
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                os.dup2(self_inner.backup, self_inner.stderr_fd)
                os.close(self_inner.backup)
                self_inner.devnull.close()
        return _Silencer()

    def _retrain_model(self) -> None:
        """
        Retrain SPM with robust fallback:
        - coverage from _choose_char_coverage (>=0.98)
        - on failure, grow vocab_size (required+margin), do NOT push coverage below 0.98
        """
        if not os.path.exists(self.data_file):
            print("⚠️ No training data available")
            return

        coverage = self._choose_char_coverage(self.current_vocab_size)

        with self._silence_cpp_logs():
            try:
                self._train_once(self.current_vocab_size, coverage)
            except Exception as e:
                msg = str(e)
                import re
                if "Vocabulary size is smaller than required_chars" in msg:
                    # Parse numbers like "... required_chars. 275 vs 293 ..."
                    m = re.search(r"required_chars\.\s*(\d+)\s*vs\s*(\d+)", msg)
                    if m:
                        n1, n2 = int(m.group(1)), int(m.group(2))
                        required = max(n1, n2)
                        new_size = max(self.current_vocab_size, required + 64)
                        print(f"🔧 Increasing vocab_size {self.current_vocab_size} → {new_size} (required ≈ {required})")
                        self.current_vocab_size = new_size
                        self.stats["current_vocab_size"] = self.current_vocab_size
                        self._train_once(self.current_vocab_size, coverage)
                    else:
                        # if we can't parse, try a reasonable vocab bump
                        new_size = max(self.current_vocab_size, int(self.current_vocab_size * 1.3))
                        print(f"🔧 Bumping vocab_size {self.current_vocab_size} → {new_size}")
                        self.current_vocab_size = new_size
                        self.stats["current_vocab_size"] = self.current_vocab_size
                        self._train_once(self.current_vocab_size, coverage)
                else:
                    # Generic failure: try increasing vocab a bit; keep coverage >= 0.98
                    new_size = max(self.current_vocab_size, int(self.current_vocab_size * 1.2))
                    print(f"🔧 Retry with larger vocab_size {self.current_vocab_size} → {new_size} (keep char_coverage={coverage})")
                    self.current_vocab_size = new_size
                    self.stats["current_vocab_size"] = self.current_vocab_size
                    self._train_once(self.current_vocab_size, coverage)

        self._load_model_silently()
        self._save_stats()
        actual = self.sp.get_piece_size() if self.sp is not None else -1
        # print(f"✅ Model retrained! target_vocab={self.current_vocab_size}, actual_vocab={actual}, char_coverage={coverage}")

    def _load_model_silently(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)


# -----------------------------
# Self-test
# -----------------------------
if __name__ == "__main__":
    tokenizer = DynamicSPTokenizer(base_vocab_size=500)

    test_texts = [
        "Hello world!",
        "你好世界！",
        "This is a test.",
        "这是一个测试。",
        "Machine learning is amazing!",
        "人工智能很棒！",
        "今日は雨です。",  # Japanese
        "🙂 Emojis! 🎉🔥",
    ]

    print("=== Testing Dynamic SentencePiece Tokenizer (robust) ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Adding text {i}: {text} ---")
        tokenizer.add_text(text)
        if tokenizer.sp is not None:
            ids = tokenizer.encode(text)
            dec = tokenizer.decode(ids)
            print(f"Encoded IDs: {ids}")
            print(f"Decoded: {dec}")
        stats = tokenizer.get_stats()
        print(f"Stats: vocab_target={stats['current_vocab_target']} | actual={stats.get('actual_vocab_size', 'N/A')} | texts_seen={stats['total_texts_seen']}")

    print("\n=== Final Stats ===")
    final_stats = tokenizer.get_stats()
    for k, v in final_stats.items():
        print(f"{k}: {v}")

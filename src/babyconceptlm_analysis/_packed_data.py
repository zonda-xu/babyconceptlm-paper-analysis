from __future__ import annotations

import hashlib
import json
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


UPOS_LABELS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
UPOS_LABEL_TO_ID = {label: idx for idx, label in enumerate(UPOS_LABELS)}


def _load_serialized_document_collection(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("format") == "sharded_tensor_documents_v1":
        chunk_dir = Path(path).resolve().parent / payload["chunk_dir"]
        shards = payload.get("shards", [])
        documents: List[torch.Tensor] = []
        for shard in shards:
            shard_path = chunk_dir / shard["file"]
            shard_payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            if not isinstance(shard_payload, (list, tuple)):
                raise TypeError(f"Expected shard {str(shard_path)!r} to contain a list/tuple of documents.")
            documents.extend(shard_payload)
        return documents
    return payload


class PackedTokenizedDataset(Dataset):
    """
    Packs tokenized documents into a single stream with <eos> separators, then
    slices that stream into fixed-length sequences.

    The final partial sequence is padded, and we optionally append a few
    eos-only padding sequences so the number of samples is divisible by the DDP
    world size. That keeps epoch accounting exact without sampler duplication.
    """

    def __init__(
        self,
        path: str,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int,
        language_id: int = 0,
        pad_to_num_sequences_multiple: Optional[int] = None,
        syntax_path: Optional[str] = None,
        word_boundary_path: Optional[str] = None,
        word_count_path: Optional[str] = None,
        word_count_multiplier: float = 1.0,
        require_word_counts: bool = False,
        syntax_pad_label: int = -100,
    ):
        super().__init__()
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")

        self.seq_length = int(seq_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.language_id = int(language_id)
        self.language_id_tensor = torch.tensor(self.language_id, dtype=torch.long)
        self.word_count_multiplier = float(word_count_multiplier)
        self.syntax_pad_label = int(syntax_pad_label)

        raw_documents = _load_serialized_document_collection(path)
        raw_syntax_documents = _load_serialized_document_collection(syntax_path) if syntax_path is not None else None
        raw_word_boundary_documents = (
            _load_serialized_document_collection(word_boundary_path) if word_boundary_path is not None else None
        )
        raw_word_count_documents = (
            _load_serialized_document_collection(word_count_path) if word_count_path is not None else None
        )
        if require_word_counts and raw_word_count_documents is None:
            raise ValueError(
                "Auditable exposure checkpoints require a token-aligned word-count sidecar. "
                f"Pass word_count_path for {path!r} and regenerate preprocessing outputs if necessary."
            )
        if raw_syntax_documents is not None and len(raw_syntax_documents) != len(raw_documents):
            raise ValueError(
                f"syntax_path={syntax_path!r} has {len(raw_syntax_documents)} documents, "
                f"but token data {path!r} has {len(raw_documents)} documents."
            )
        if raw_word_boundary_documents is not None and len(raw_word_boundary_documents) != len(raw_documents):
            raise ValueError(
                f"word_boundary_path={word_boundary_path!r} has {len(raw_word_boundary_documents)} documents, "
                f"but token data {path!r} has {len(raw_documents)} documents."
            )
        if raw_word_count_documents is not None and len(raw_word_count_documents) != len(raw_documents):
            raise ValueError(
                f"word_count_path={word_count_path!r} has {len(raw_word_count_documents)} documents, "
                f"but token data {path!r} has {len(raw_documents)} documents."
            )
        normalized_docs: List[torch.Tensor] = []
        normalized_syntax_docs: List[torch.Tensor] = []
        normalized_word_boundary_docs: List[torch.Tensor] = []
        normalized_word_count_docs: List[torch.Tensor] = []
        for idx, doc in enumerate(raw_documents):
            if not isinstance(doc, torch.Tensor):
                doc = torch.tensor(doc, dtype=torch.long)
            else:
                doc = doc.long()
            if doc.dim() != 1:
                doc = doc.reshape(-1)
            syntax_doc = None
            if raw_syntax_documents is not None:
                syntax_doc = raw_syntax_documents[idx]
                if not isinstance(syntax_doc, torch.Tensor):
                    syntax_doc = torch.tensor(syntax_doc, dtype=torch.long)
                else:
                    syntax_doc = syntax_doc.long()
                if syntax_doc.dim() != 1:
                    syntax_doc = syntax_doc.reshape(-1)
                if syntax_doc.numel() != doc.numel():
                    raise ValueError(
                        f"Document {idx} token count mismatch between {path!r} ({doc.numel()}) "
                        f"and {syntax_path!r} ({syntax_doc.numel()})."
                    )
            word_boundary_doc = None
            if raw_word_boundary_documents is not None:
                word_boundary_doc = raw_word_boundary_documents[idx]
                if not isinstance(word_boundary_doc, torch.Tensor):
                    word_boundary_doc = torch.tensor(word_boundary_doc, dtype=torch.long)
                else:
                    word_boundary_doc = word_boundary_doc.long()
                if word_boundary_doc.dim() != 1:
                    word_boundary_doc = word_boundary_doc.reshape(-1)
                if word_boundary_doc.numel() != doc.numel():
                    raise ValueError(
                        f"Document {idx} token count mismatch between {path!r} ({doc.numel()}) "
                        f"and {word_boundary_path!r} ({word_boundary_doc.numel()})."
                    )
            word_count_doc = None
            if raw_word_count_documents is not None:
                word_count_doc = raw_word_count_documents[idx]
                if not isinstance(word_count_doc, torch.Tensor):
                    word_count_doc = torch.tensor(word_count_doc, dtype=torch.long)
                else:
                    word_count_doc = word_count_doc.long()
                if word_count_doc.dim() != 1:
                    word_count_doc = word_count_doc.reshape(-1)
                if word_count_doc.numel() != doc.numel():
                    raise ValueError(
                        f"Document {idx} token count mismatch between {path!r} ({doc.numel()}) "
                        f"and {word_count_path!r} ({word_count_doc.numel()})."
                    )
                if (word_count_doc < 0).any():
                    raise ValueError(f"Document {idx} in {word_count_path!r} contains negative word counts.")
            if doc.numel() > 0:
                normalized_docs.append(doc)
                if syntax_doc is not None:
                    normalized_syntax_docs.append(syntax_doc)
                if word_boundary_doc is not None:
                    normalized_word_boundary_docs.append(word_boundary_doc)
                if word_count_doc is not None:
                    normalized_word_count_docs.append(word_count_doc)

        if not normalized_docs:
            raise ValueError(f"No valid token documents found in {path}")

        eos_tensor = torch.tensor([self.eos_token_id], dtype=torch.long)
        syntax_eos_tensor = torch.tensor([self.syntax_pad_label], dtype=torch.long)
        packed_parts: List[torch.Tensor] = []
        packed_syntax_parts: List[torch.Tensor] = []
        packed_word_boundary_parts: List[torch.Tensor] = []
        packed_word_count_parts: List[torch.Tensor] = []
        for idx, doc in enumerate(normalized_docs):
            packed_parts.append(doc)
            if normalized_syntax_docs:
                packed_syntax_parts.append(normalized_syntax_docs[idx])
            if normalized_word_boundary_docs:
                packed_word_boundary_parts.append(normalized_word_boundary_docs[idx])
            if normalized_word_count_docs:
                packed_word_count_parts.append(normalized_word_count_docs[idx])
            if idx != len(normalized_docs) - 1:
                if int(doc[-1].item()) != self.eos_token_id:
                    packed_parts.append(eos_tensor)
                    if normalized_syntax_docs:
                        packed_syntax_parts.append(syntax_eos_tensor)
                    if normalized_word_boundary_docs:
                        packed_word_boundary_parts.append(syntax_eos_tensor)
                    if normalized_word_count_docs:
                        packed_word_count_parts.append(torch.zeros(1, dtype=torch.long))

        packed_stream = torch.cat(packed_parts, dim=0)
        packed_syntax_stream = torch.cat(packed_syntax_parts, dim=0) if packed_syntax_parts else None
        packed_word_boundary_stream = (
            torch.cat(packed_word_boundary_parts, dim=0) if packed_word_boundary_parts else None
        )
        packed_word_count_stream = torch.cat(packed_word_count_parts, dim=0) if packed_word_count_parts else None
        self.real_token_count = int(packed_stream.numel())
        self.num_documents = len(normalized_docs)

        real_num_sequences = max(1, (self.real_token_count + self.seq_length - 1) // self.seq_length)
        total_token_slots = real_num_sequences * self.seq_length

        input_buffer = torch.full((total_token_slots,), fill_value=self.pad_token_id, dtype=torch.long)
        attention_buffer = torch.zeros((total_token_slots,), dtype=torch.long)
        input_buffer[: self.real_token_count] = packed_stream
        attention_buffer[: self.real_token_count] = 1
        syntax_buffer = None
        if packed_syntax_stream is not None:
            syntax_buffer = torch.full((total_token_slots,), fill_value=self.syntax_pad_label, dtype=torch.long)
            syntax_buffer[: self.real_token_count] = packed_syntax_stream
        word_boundary_buffer = None
        if packed_word_boundary_stream is not None:
            word_boundary_buffer = torch.full((total_token_slots,), fill_value=self.syntax_pad_label, dtype=torch.long)
            word_boundary_buffer[: self.real_token_count] = packed_word_boundary_stream
        word_count_buffer = None
        if packed_word_count_stream is not None:
            word_count_buffer = torch.zeros((total_token_slots,), dtype=torch.long)
            word_count_buffer[: self.real_token_count] = packed_word_count_stream

        num_sequences = real_num_sequences
        self.num_real_sequences = real_num_sequences
        multiple = max(1, int(pad_to_num_sequences_multiple or 1))
        remainder = num_sequences % multiple
        if remainder != 0:
            extra_sequences = multiple - remainder
            extra_input = torch.full((extra_sequences, self.seq_length), fill_value=self.pad_token_id, dtype=torch.long)
            extra_attention = torch.zeros((extra_sequences, self.seq_length), dtype=torch.long)
            extra_input[:, 0] = self.eos_token_id
            extra_attention[:, 0] = 1
            extra_syntax = None
            if syntax_buffer is not None:
                extra_syntax = torch.full(
                    (extra_sequences, self.seq_length),
                    fill_value=self.syntax_pad_label,
                    dtype=torch.long,
                )
            extra_word_boundary = None
            if word_boundary_buffer is not None:
                extra_word_boundary = torch.full(
                    (extra_sequences, self.seq_length),
                    fill_value=self.syntax_pad_label,
                    dtype=torch.long,
                )
            extra_word_counts = None
            if word_count_buffer is not None:
                extra_word_counts = torch.zeros((extra_sequences, self.seq_length), dtype=torch.long)

            self.input_ids = torch.cat([input_buffer.view(real_num_sequences, self.seq_length), extra_input], dim=0)
            self.attention_mask = torch.cat([attention_buffer.view(real_num_sequences, self.seq_length), extra_attention], dim=0)
            self.syntax_labels = (
                torch.cat([syntax_buffer.view(real_num_sequences, self.seq_length), extra_syntax], dim=0)
                if syntax_buffer is not None
                else None
            )
            self.word_boundary_labels = (
                torch.cat([word_boundary_buffer.view(real_num_sequences, self.seq_length), extra_word_boundary], dim=0)
                if word_boundary_buffer is not None
                else None
            )
            self.word_counts = (
                torch.cat([word_count_buffer.view(real_num_sequences, self.seq_length), extra_word_counts], dim=0)
                if word_count_buffer is not None
                else None
            )
        else:
            self.input_ids = input_buffer.view(real_num_sequences, self.seq_length)
            self.attention_mask = attention_buffer.view(real_num_sequences, self.seq_length)
            self.syntax_labels = syntax_buffer.view(real_num_sequences, self.seq_length) if syntax_buffer is not None else None
            self.word_boundary_labels = (
                word_boundary_buffer.view(real_num_sequences, self.seq_length)
                if word_boundary_buffer is not None
                else None
            )
            self.word_counts = (
                word_count_buffer.view(real_num_sequences, self.seq_length)
                if word_count_buffer is not None
                else None
            )

        self.num_sequences = int(self.input_ids.size(0))
        self.raw_word_count = int(self.word_counts.sum().item()) if self.word_counts is not None else None
        self.adjusted_word_count = (
            float(self.raw_word_count) * self.word_count_multiplier if self.raw_word_count is not None else None
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "language_id": self.language_id_tensor,
        }
        if self.syntax_labels is not None:
            item["syntax_labels"] = self.syntax_labels[idx]
        if self.word_boundary_labels is not None:
            item["word_boundary_labels"] = self.word_boundary_labels[idx]
        if self.word_counts is not None:
            raw_word_count = self.word_counts[idx].sum()
            item["raw_word_count"] = raw_word_count
            item["adjusted_word_count"] = raw_word_count.to(torch.float64) * self.word_count_multiplier
        return item


class MultiPackedTokenizedDataset(Dataset):
    """
    Manifest-driven multilingual dataset.

    Each language is packed independently through PackedTokenizedDataset, so a
    fixed-length sequence never crosses language boundaries. The collator keeps
    a scalar language_id per sequence and later expands it to token positions.
    Manifest format:
      {
        "format": "multi1_multilingual_manifest_v1",
        "languages": [
          {"name": "eng", "id": 0, "train_path": "...", "validation_path": "..."}
        ]
      }
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int,
        pad_to_num_sequences_multiple: Optional[int] = None,
        require_word_counts: bool = False,
        syntax_pad_label: int = -100,
    ):
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        manifest_file = Path(manifest_path)
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "multi1_multilingual_manifest_v1":
            raise ValueError(f"Unsupported multilingual manifest format in {manifest_path!r}.")

        path_key = "train_path" if self.split == "train" else "validation_path"
        root = manifest_file.resolve().parent
        self.datasets: List[PackedTokenizedDataset] = []
        self.language_names: List[str] = []
        self.language_exposure_accounting: List[Dict[str, object]] = []
        self.cumulative_lengths: List[int] = []
        total_len = 0
        total_docs = 0
        total_tokens = 0
        total_raw_words = 0
        total_adjusted_words = 0.0

        for entry in manifest.get("languages", []):
            if path_key not in entry:
                continue
            data_path = Path(entry[path_key])
            if not data_path.is_absolute():
                data_path = root / data_path
            word_count_key = "train_word_count_path" if self.split == "train" else "validation_word_count_path"
            word_count_path = entry.get(word_count_key)
            if word_count_path is not None:
                word_count_path = Path(word_count_path)
                if not word_count_path.is_absolute():
                    word_count_path = root / word_count_path
            premium_value = entry.get("budget", {}).get("byte_premium")
            if require_word_counts and premium_value is None:
                raise ValueError(
                    f"Manifest language {entry.get('name', entry['id'])!r} lacks budget.byte_premium; "
                    "auditable multilingual exposure cannot assume a default."
                )
            byte_premium = float(1.0 if premium_value is None else premium_value)
            if byte_premium <= 0:
                raise ValueError(
                    f"Manifest language {entry.get('name', entry['id'])!r} has invalid Byte Premium {byte_premium}."
                )
            dataset = PackedTokenizedDataset(
                path=str(data_path),
                seq_length=seq_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                language_id=int(entry["id"]),
                pad_to_num_sequences_multiple=pad_to_num_sequences_multiple,
                syntax_path=None,
                word_boundary_path=None,
                word_count_path=str(word_count_path) if word_count_path is not None else None,
                word_count_multiplier=byte_premium,
                require_word_counts=require_word_counts,
                syntax_pad_label=syntax_pad_label,
            )
            self.datasets.append(dataset)
            self.language_names.append(str(entry.get("name", entry["id"])))
            total_len += len(dataset)
            total_docs += int(dataset.num_documents)
            total_tokens += int(dataset.real_token_count)
            if dataset.raw_word_count is not None:
                total_raw_words += int(dataset.raw_word_count)
                total_adjusted_words += float(dataset.adjusted_word_count)
            self.language_exposure_accounting.append(
                {
                    "name": str(entry.get("name", entry["id"])),
                    "language_id": int(entry["id"]),
                    "byte_premium": byte_premium,
                    "raw_words": dataset.raw_word_count,
                    "adjusted_words": dataset.adjusted_word_count,
                    "word_count_path": str(word_count_path) if word_count_path is not None else None,
                }
            )
            self.cumulative_lengths.append(total_len)

        if not self.datasets:
            raise ValueError(f"No datasets found for split={self.split!r} in {manifest_path!r}.")

        self.seq_length = int(seq_length)
        self.num_sequences = int(total_len)
        self.num_real_sequences = int(sum(ds.num_real_sequences for ds in self.datasets))
        self.num_documents = int(total_docs)
        self.real_token_count = int(total_tokens)
        self.raw_word_count = int(total_raw_words) if all(ds.raw_word_count is not None for ds in self.datasets) else None
        self.adjusted_word_count = (
            float(total_adjusted_words) if all(ds.adjusted_word_count is not None for ds in self.datasets) else None
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][idx - prev]


@dataclass
class PackedSequenceCollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([feat["input_ids"] for feat in features], dim=0),
            "attention_mask": torch.stack([feat["attention_mask"] for feat in features], dim=0),
            "language_ids": torch.stack([feat["language_id"] for feat in features], dim=0),
        }
        if "syntax_labels" in features[0]:
            batch["syntax_labels"] = torch.stack([feat["syntax_labels"] for feat in features], dim=0)
        if "word_boundary_labels" in features[0]:
            batch["word_boundary_labels"] = torch.stack([feat["word_boundary_labels"] for feat in features], dim=0)
        if "raw_word_count" in features[0]:
            batch["raw_word_counts"] = torch.stack([feat["raw_word_count"] for feat in features], dim=0)
            batch["adjusted_word_counts"] = torch.stack([feat["adjusted_word_count"] for feat in features], dim=0)
        return batch

#!/usr/bin/env python3
"""Run reproducible held-out segmentation analysis for BabyConceptLM checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    language_id: int = 0
    word_boundary_path: Optional[Path] = None
    syntax_path: Optional[Path] = None
    word_count_path: Optional[Path] = None


@dataclass
class SampledDocument:
    source_index: int
    tokens: torch.Tensor
    word_boundaries: Optional[torch.Tensor] = None
    syntax_labels: Optional[torch.Tensor] = None
    word_counts: Optional[torch.Tensor] = None


@dataclass
class AnalysisWindow:
    document_slot: int
    source_index: int
    tokens: torch.Tensor
    language_id: int
    left_censored: bool
    right_censored: bool
    word_boundaries: Optional[torch.Tensor] = None
    syntax_labels: Optional[torch.Tensor] = None
    word_counts: Optional[torch.Tensor] = None


@dataclass
class DocumentStats:
    source_index: int
    windows: int = 0
    tokens: int = 0
    concepts: int = 0
    complete_segments: int = 0
    length_histogram: Counter = field(default_factory=Counter)
    complete_length_histogram: Counter = field(default_factory=Counter)
    boundary_eligible: int = 0
    observed_boundaries: int = 0
    learned_boundaries: int = 0
    max_forced_boundaries: int = 0
    boundary_probability_sum: float = 0.0
    boundary_probability_count: int = 0
    hard_boundary_probability_sum: float = 0.0
    hard_boundary_probability_count: int = 0
    interior_boundary_probability_sum: float = 0.0
    interior_boundary_probability_count: int = 0
    gate_sum: float = 0.0
    gate_square_sum: float = 0.0
    gate_count: int = 0
    boundary_gate_sum: float = 0.0
    boundary_gate_count: int = 0
    interior_gate_sum: float = 0.0
    interior_gate_count: int = 0
    word_tp: int = 0
    word_fp: int = 0
    word_fn: int = 0
    word_tn: int = 0
    word_start_positions: int = 0
    word_inside_positions: int = 0
    boundaries_at_word_starts: int = 0
    boundaries_inside_words: int = 0
    syntax_change_positions: int = 0
    syntax_same_positions: int = 0
    boundaries_at_syntax_changes: int = 0
    boundaries_at_syntax_same: int = 0
    segment_pos_purity_sum: float = 0.0
    segment_pos_entropy_sum: float = 0.0
    segment_pos_count: int = 0
    source_words: float = 0.0
    decoded_nonspace_characters: int = 0
    decoded_whitespace_units: int = 0
    decoded_segment_count: int = 0


@dataclass
class DatasetAccumulator:
    documents: Dict[int, DocumentStats] = field(default_factory=dict)
    boundary_probabilities: List[np.ndarray] = field(default_factory=list)
    hard_boundary_labels: List[np.ndarray] = field(default_factory=list)
    gate_values: List[np.ndarray] = field(default_factory=list)
    gate_transition_values: List[np.ndarray] = field(default_factory=list)
    gate_boundary_probabilities: List[np.ndarray] = field(default_factory=list)
    word_boundary_probabilities: List[np.ndarray] = field(default_factory=list)
    word_boundary_labels: List[np.ndarray] = field(default_factory=list)
    examples: List[Dict[str, object]] = field(default_factory=list)


class JsonBackedTokenizer:
    def __init__(
        self,
        source_path: Path,
        tokenizer_json: Mapping[str, object],
        tokenizer_config: Optional[Mapping[str, object]] = None,
        special_tokens_map: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.source_path = source_path
        self.source_dir = source_path if source_path.is_dir() else source_path.parent
        vocab = tokenizer_json["model"]["vocab"]
        self.token_to_id = {str(token): int(index) for token, index in vocab.items()}
        self.id_to_token = {int(index): str(token) for token, index in vocab.items()}
        config = dict(tokenizer_config or {})
        special_map = dict(special_tokens_map or {})
        self.bos_token = str(special_map.get("bos_token", config.get("bos_token", "<s>")))
        self.eos_token = str(special_map.get("eos_token", config.get("eos_token", "</s>")))
        self.unk_token = str(special_map.get("unk_token", config.get("unk_token", "<unk>")))
        self.pad_token = str(special_map.get("pad_token", config.get("pad_token", "<pad>")))
        self.mask_token = str(special_map.get("mask_token", config.get("mask_token", "<mask>")))
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.all_special_ids = [
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self.mask_token_id,
        ]
        decoder = tokenizer_json.get("decoder") or {}
        self.decoder_type = str(decoder.get("type", "identity"))
        self.metaspace_replacement = str(decoder.get("replacement", "▁"))
        self.byte_decoder = self._build_byte_decoder() if self.decoder_type == "ByteLevel" else {}

    @staticmethod
    def _build_byte_decoder() -> Dict[str, int]:
        byte_values = list(range(ord("!"), ord("~") + 1))
        byte_values += list(range(ord("¡"), ord("¬") + 1))
        byte_values += list(range(ord("®"), ord("ÿ") + 1))
        code_points = list(byte_values)
        extra = 0
        for byte_value in range(256):
            if byte_value in byte_values:
                continue
            byte_values.append(byte_value)
            code_points.append(256 + extra)
            extra += 1
        return {chr(code_point): byte_value for byte_value, code_point in zip(byte_values, code_points)}

    def __len__(self) -> int:
        return len(self.id_to_token)

    def convert_ids_to_tokens(self, index: int) -> str:
        return self.id_to_token.get(int(index), self.unk_token)

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> str:
        del clean_up_tokenization_spaces, kwargs
        special_ids = set(self.all_special_ids) if skip_special_tokens else set()
        tokens = [self.convert_ids_to_tokens(token_id) for token_id in token_ids if int(token_id) not in special_ids]
        if self.decoder_type == "ByteLevel":
            encoded = "".join(tokens)
            byte_values = bytearray(self.byte_decoder[character] for character in encoded if character in self.byte_decoder)
            return byte_values.decode("utf-8", errors="replace")
        if self.decoder_type == "Metaspace":
            return "".join(tokens).replace(self.metaspace_replacement, " ")
        if self.decoder_type == "WordPiece":
            text = ""
            for token in tokens:
                if token.startswith("##"):
                    text += token[2:]
                else:
                    text += (" " if text else "") + token
            return text
        return "".join(tokens)

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> List[str]:
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for sequence in sequences
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze BabyConceptLM segment lengths, learned/forced boundaries, optional linguistic "
            "alignment, multilingual compression, and the bounded LM residual gate."
        )
    )
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--tokenizer_path", default=None, type=str)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--language_id", action="append", default=[], metavar="NAME=ID")
    parser.add_argument("--word_boundary_labels", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--syntax_labels", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--word_counts", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--manifest", default=None, type=str)
    parser.add_argument("--manifest_split", default="validation", choices=("train", "validation"))
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--seq_length", default=256, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_documents", default=2000, type=int)
    parser.add_argument("--document_sampling", default="reservoir", choices=("reservoir", "first"))
    parser.add_argument("--windows_per_document", default=1, type=int)
    parser.add_argument("--bootstrap_samples", default=1000, type=int)
    parser.add_argument("--confidence", default=0.95, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--num_examples", default=0, type=int)
    parser.add_argument("--skip_decode_metrics", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--accept_remote_code", action="store_true", help="Allow execution of reviewed local checkpoint Python code.")
    parser.add_argument("--hf_modules_cache", default=None, type=str)
    parser.add_argument("--report_to", default="none", choices=("none", "wandb"))
    parser.add_argument("--wandb_project", default="babyconceptlm-segmentation", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    return parser.parse_args()


def parse_named_values(
    values: Sequence[str],
    argument_name: str,
    converter,
) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument_name} expects NAME=VALUE, got {value!r}.")
        name, raw_value = value.split("=", 1)
        name = name.strip()
        if not name or not raw_value:
            raise ValueError(f"{argument_name} expects non-empty NAME=VALUE, got {value!r}.")
        if name in parsed:
            raise ValueError(f"Duplicate {argument_name} entry for {name!r}.")
        parsed[name] = converter(raw_value)
    return parsed


def collect_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    entries = parse_named_values(args.model, "--model", lambda value: Path(value).expanduser().resolve())
    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        model_name = args.model_name or model_path.name
        if model_name in entries:
            raise ValueError(f"Duplicate model name {model_name!r}.")
        entries[model_name] = model_path
    if not entries:
        raise ValueError("Pass at least one --model NAME=PATH or --model_path PATH.")
    specs = [ModelSpec(name=name, path=Path(path)) for name, path in entries.items()]
    for spec in specs:
        if not spec.path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {spec.path}")
        if not (spec.path / "config.json").is_file():
            raise FileNotFoundError(f"Model directory lacks config.json: {spec.path}")
    return specs


def resolve_manifest_path(raw_path: Optional[str], manifest_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    attempts = [candidate]
    if not candidate.is_absolute():
        attempts.append(manifest_dir / candidate)
    else:
        attempts.append(manifest_dir / candidate.name)
    for attempt in attempts:
        resolved = attempt.resolve()
        if resolved.exists():
            return resolved
    return attempts[-1].resolve()


def dataset_specs_from_manifest(path: Path, split: str) -> List[DatasetSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ValueError(f"Manifest contains no languages: {path}")
    specs: List[DatasetSpec] = []
    for language in languages:
        name = str(language["name"])
        split_payload = language.get(split, {})
        data_path = split_payload.get("path") or language.get(f"{split}_path")
        word_count_path = split_payload.get("word_count_path") or language.get(f"{split}_word_count_path")
        resolved_data = resolve_manifest_path(data_path, path.parent)
        if resolved_data is None:
            raise ValueError(f"Manifest language {name!r} lacks a {split} path.")
        specs.append(
            DatasetSpec(
                name=name,
                path=resolved_data,
                language_id=int(language.get("id", 0)),
                word_count_path=resolve_manifest_path(word_count_path, path.parent),
            )
        )
    return specs


def collect_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    specs: Dict[str, DatasetSpec] = {}
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        for spec in dataset_specs_from_manifest(manifest_path, args.manifest_split):
            specs[spec.name] = spec

    data_paths = parse_named_values(args.dataset, "--dataset", lambda value: Path(value).expanduser().resolve())
    language_ids = parse_named_values(args.language_id, "--language_id", int)
    word_boundaries = parse_named_values(
        args.word_boundary_labels,
        "--word_boundary_labels",
        lambda value: Path(value).expanduser().resolve(),
    )
    syntax_labels = parse_named_values(
        args.syntax_labels,
        "--syntax_labels",
        lambda value: Path(value).expanduser().resolve(),
    )
    word_counts = parse_named_values(
        args.word_counts,
        "--word_counts",
        lambda value: Path(value).expanduser().resolve(),
    )

    known_names = set(specs) | set(data_paths)
    for mapping_name, mapping in (
        ("--language_id", language_ids),
        ("--word_boundary_labels", word_boundaries),
        ("--syntax_labels", syntax_labels),
        ("--word_counts", word_counts),
    ):
        unknown = set(mapping) - known_names
        if unknown:
            raise ValueError(f"{mapping_name} refers to unknown datasets: {sorted(unknown)}")

    for name in known_names:
        previous = specs.get(name)
        data_path = Path(data_paths.get(name, previous.path if previous else ""))
        specs[name] = DatasetSpec(
            name=name,
            path=data_path,
            language_id=int(language_ids.get(name, previous.language_id if previous else 0)),
            word_boundary_path=Path(
                word_boundaries.get(name, previous.word_boundary_path if previous else "")
            )
            if word_boundaries.get(name, previous.word_boundary_path if previous else None)
            else None,
            syntax_path=Path(syntax_labels.get(name, previous.syntax_path if previous else ""))
            if syntax_labels.get(name, previous.syntax_path if previous else None)
            else None,
            word_count_path=Path(word_counts.get(name, previous.word_count_path if previous else ""))
            if word_counts.get(name, previous.word_count_path if previous else None)
            else None,
        )

    if not specs:
        raise ValueError("Pass --manifest or at least one --dataset NAME=PATH.")
    for spec in specs.values():
        for label, candidate in (
            ("dataset", spec.path),
            ("word-boundary labels", spec.word_boundary_path),
            ("syntax labels", spec.syntax_path),
            ("word counts", spec.word_count_path),
        ):
            if candidate is not None and not candidate.is_file():
                raise FileNotFoundError(f"Missing {label} for {spec.name}: {candidate}")
    return list(specs.values())


def safe_torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def normalize_document(value, dtype: torch.dtype = torch.long) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().to(dtype=dtype)
    else:
        tensor = torch.tensor(value, dtype=dtype)
    return tensor.reshape(-1)


def iter_serialized_documents(path: Path) -> Iterator[torch.Tensor]:
    payload = safe_torch_load(path)
    if isinstance(payload, dict) and payload.get("format") == "sharded_tensor_documents_v1":
        chunk_dir = path.resolve().parent / payload["chunk_dir"]
        for shard in payload.get("shards", []):
            shard_payload = safe_torch_load(chunk_dir / shard["file"])
            if not isinstance(shard_payload, (list, tuple)):
                raise TypeError(f"Shard {shard['file']!r} in {path} is not a document list.")
            for document in shard_payload:
                yield normalize_document(document)
        return
    if isinstance(payload, torch.Tensor) and payload.dim() == 2:
        for document in payload:
            yield normalize_document(document)
        return
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"Expected {path} to contain a document collection, got {type(payload).__name__}.")
    for document in payload:
        yield normalize_document(document)


def paired_documents(spec: DatasetSpec) -> Iterator[SampledDocument]:
    sentinel = object()
    iterators: List[Iterable] = [iter_serialized_documents(spec.path)]
    optional_paths = [spec.word_boundary_path, spec.syntax_path, spec.word_count_path]
    for optional_path in optional_paths:
        iterators.append(iter_serialized_documents(optional_path) if optional_path is not None else [])

    if all(path is not None for path in optional_paths):
        combined = zip_longest(*iterators, fillvalue=sentinel)
    else:
        token_iterator = iterators[0]
        optional_iterators = [iter(value) if path is not None else None for value, path in zip(iterators[1:], optional_paths)]

        def generated():
            for token_document in token_iterator:
                values = [token_document]
                for optional_iterator in optional_iterators:
                    values.append(next(optional_iterator, sentinel) if optional_iterator is not None else None)
                yield tuple(values)
            for optional_iterator in optional_iterators:
                if optional_iterator is not None and next(optional_iterator, sentinel) is not sentinel:
                    yield (sentinel, sentinel, sentinel, sentinel)

        combined = generated()

    for source_index, values in enumerate(combined):
        tokens, word_boundaries, syntax_labels, word_counts = values
        if tokens is sentinel or any(value is sentinel for value in values[1:] if value is not None):
            raise ValueError(f"Document count mismatch among data and sidecars for {spec.name}.")
        tokens = normalize_document(tokens)
        optional_values = []
        for label, value in (
            ("word boundaries", word_boundaries),
            ("syntax labels", syntax_labels),
            ("word counts", word_counts),
        ):
            if value is None:
                optional_values.append(None)
                continue
            tensor = normalize_document(value)
            if tensor.numel() != tokens.numel():
                raise ValueError(
                    f"{spec.name} document {source_index} has {tokens.numel()} tokens but "
                    f"{tensor.numel()} {label}."
                )
            optional_values.append(tensor)
        yield SampledDocument(
            source_index=source_index,
            tokens=tokens,
            word_boundaries=optional_values[0],
            syntax_labels=optional_values[1],
            word_counts=optional_values[2],
        )


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_documents(
    spec: DatasetSpec,
    max_documents: int,
    sampling: str,
    seed: int,
) -> Tuple[List[SampledDocument], int]:
    limit = None if max_documents <= 0 else int(max_documents)
    selected: List[SampledDocument] = []
    scanned = 0
    rng = random.Random(stable_seed(seed, spec.name))
    for document in paired_documents(spec):
        scanned += 1
        if document.tokens.numel() == 0:
            continue
        if limit is None or len(selected) < limit:
            selected.append(document)
        elif sampling == "first":
            break
        else:
            replacement = rng.randrange(scanned)
            if replacement < limit:
                selected[replacement] = document
    selected.sort(key=lambda item: item.source_index)
    return selected, scanned


def content_spans(tokens: torch.Tensor, separator_ids: Sequence[int]) -> List[Tuple[int, int]]:
    separators = set(int(value) for value in separator_ids if value is not None)
    spans: List[Tuple[int, int]] = []
    start = 0
    for index, token_id in enumerate(tokens.tolist()):
        if int(token_id) not in separators:
            continue
        if start < index:
            spans.append((start, index))
        start = index + 1
    if start < tokens.numel():
        spans.append((start, int(tokens.numel())))
    return spans


def map_candidate_index(
    spans: Sequence[Tuple[int, int]],
    seq_length: int,
    candidate_index: int,
) -> Tuple[int, int, bool, bool]:
    remaining = int(candidate_index)
    for span_start, span_end in spans:
        span_length = span_end - span_start
        choices = max(1, span_length - seq_length + 1)
        if remaining >= choices:
            remaining -= choices
            continue
        local_start = span_start + remaining if span_length > seq_length else span_start
        local_end = min(span_end, local_start + seq_length)
        return local_start, local_end, local_start > span_start, local_end < span_end
    raise IndexError(f"Candidate index {candidate_index} is outside the available windows.")


def window_candidates(
    document: SampledDocument,
    seq_length: int,
    windows_per_document: int,
    seed: int,
    separator_ids: Sequence[int],
) -> List[Tuple[int, int, bool, bool]]:
    spans = content_spans(document.tokens, separator_ids)
    if not spans:
        return []
    if windows_per_document == 0:
        windows: List[Tuple[int, int, bool, bool]] = []
        for span_start, span_end in spans:
            for local_start in range(span_start, span_end, seq_length):
                local_end = min(span_end, local_start + seq_length)
                windows.append((local_start, local_end, local_start > span_start, local_end < span_end))
        return windows

    total_candidates = sum(max(1, span_end - span_start - seq_length + 1) for span_start, span_end in spans)
    requested = min(max(1, windows_per_document), total_candidates)
    rng = random.Random(stable_seed(seed ^ document.source_index, str(document.source_index)))
    candidate_ids = sorted(rng.sample(range(total_candidates), requested))
    return [map_candidate_index(spans, seq_length, candidate_id) for candidate_id in candidate_ids]


def build_windows(
    documents: Sequence[SampledDocument],
    spec: DatasetSpec,
    seq_length: int,
    windows_per_document: int,
    seed: int,
    separator_ids: Sequence[int],
) -> List[AnalysisWindow]:
    windows: List[AnalysisWindow] = []
    for document_slot, document in enumerate(documents):
        for start, end, left_censored, right_censored in window_candidates(
            document,
            seq_length,
            windows_per_document,
            seed,
            separator_ids,
        ):
            windows.append(
                AnalysisWindow(
                    document_slot=document_slot,
                    source_index=document.source_index,
                    tokens=document.tokens[start:end],
                    language_id=spec.language_id,
                    left_censored=left_censored,
                    right_censored=right_censored,
                    word_boundaries=document.word_boundaries[start:end]
                    if document.word_boundaries is not None
                    else None,
                    syntax_labels=document.syntax_labels[start:end]
                    if document.syntax_labels is not None
                    else None,
                    word_counts=document.word_counts[start:end] if document.word_counts is not None else None,
                )
            )
    return windows


def collate_windows(windows: Sequence[AnalysisWindow], pad_token_id: int) -> Dict[str, object]:
    max_length = max(int(window.tokens.numel()) for window in windows)
    batch_size = len(windows)
    input_ids = torch.full((batch_size, max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    language_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    syntax_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    has_syntax = any(window.syntax_labels is not None for window in windows)
    for row, window in enumerate(windows):
        length = int(window.tokens.numel())
        input_ids[row, :length] = window.tokens
        attention_mask[row, :length] = 1
        language_ids[row, :length] = int(window.language_id)
        if window.syntax_labels is not None:
            syntax_labels[row, :length] = window.syntax_labels
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "language_ids": language_ids,
        "syntax_token_labels": syntax_labels if has_syntax else None,
        "windows": list(windows),
    }


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    original_path = path
    if path.is_dir():
        try:
            return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True, use_fast=True)
        except Exception:
            tokenizer_file = path / "tokenizer.json"
            if not tokenizer_file.is_file():
                raise
            path = tokenizer_file
    try:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(path),
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>",
        )
    except Exception:
        tokenizer_payload = json.loads(path.read_text(encoding="utf-8"))
        source_dir = original_path if original_path.is_dir() else original_path.parent
        config_path = source_dir / "tokenizer_config.json"
        special_map_path = source_dir / "special_tokens_map.json"
        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
        special_tokens_map = (
            json.loads(special_map_path.read_text(encoding="utf-8")) if special_map_path.is_file() else None
        )
        return JsonBackedTokenizer(
            source_path=original_path,
            tokenizer_json=tokenizer_payload,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
        )


def resolve_device(raw_device: Optional[str]) -> torch.device:
    if raw_device:
        device = torch.device(raw_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return contextlib.nullcontext()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def compute_lm_residual_gate(model, outputs, attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
    gate_module = getattr(model, "lm_fusion_gate", None)
    if gate_module is None or not bool(getattr(model.config, "lm_concept_residual_gate", False)):
        return None
    shell_states = getattr(outputs, "shell_token_states", None)
    readout_states = getattr(outputs, "readout_raw_states", None)
    if readout_states is None:
        readout_states = getattr(outputs, "token_states", None)
    if shell_states is None or readout_states is None:
        raise RuntimeError("The checkpoint enables the LM residual gate but does not expose its input states.")
    concept_residual = readout_states - shell_states
    shell_residual = shell_states - readout_states
    feature_kind = str(getattr(model.config, "lm_concept_gate_boundary_feature", "prob")).lower()
    if feature_kind == "certainty" and getattr(outputs, "boundary_certainty", None) is not None:
        boundary_feature = outputs.boundary_certainty
    elif feature_kind == "prob" and getattr(outputs, "boundary_probs", None) is not None:
        boundary_feature = outputs.boundary_probs
    else:
        boundary_feature = shell_states.new_zeros(shell_states.shape[:2])
    boundary_feature = boundary_feature.to(dtype=shell_states.dtype).unsqueeze(-1)
    gate_mode = str(getattr(model.config, "lm_concept_gate_mode", "concept_main_shell_residual")).lower()
    if gate_mode == "shell_main_concept_residual":
        gate_input = torch.cat([shell_states, readout_states, concept_residual, boundary_feature], dim=-1)
    elif gate_mode == "concept_main_shell_residual":
        gate_input = torch.cat([readout_states, shell_states, shell_residual, boundary_feature], dim=-1)
    else:
        raise ValueError(f"Unknown lm_concept_gate_mode: {gate_mode!r}")
    gate_max = float(getattr(model.config, "lm_concept_gate_max", 0.0))
    gate = torch.sigmoid(gate_module(gate_input)) * gate_max
    return gate.squeeze(-1) * attention_mask.to(dtype=gate.dtype)


def segment_runs(segment_ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not segment_ids:
        return []
    runs: List[Tuple[int, int, int]] = []
    start = 0
    current = int(segment_ids[0])
    for index in range(1, len(segment_ids)):
        value = int(segment_ids[index])
        if value == current:
            continue
        runs.append((current, start, index))
        current = value
        start = index
    runs.append((current, start, len(segment_ids)))
    return runs


def classify_segmentation(
    segment_ids: Sequence[int],
    max_chunk_len: int,
    left_censored: bool,
    right_censored: bool,
) -> Dict[str, object]:
    runs = segment_runs(segment_ids)
    lengths = [end - start for _, start, end in runs]
    complete_lengths = list(lengths)
    if left_censored and complete_lengths:
        complete_lengths = complete_lengths[1:]
    if right_censored and complete_lengths:
        complete_lengths = complete_lengths[:-1]
    boundary_positions = [start for _, start, _ in runs[1:]]
    max_forced_positions = []
    for position, previous_length in zip(boundary_positions, lengths[:-1]):
        if previous_length >= max_chunk_len:
            max_forced_positions.append(position)
    max_forced = set(max_forced_positions)
    learned_positions = [position for position in boundary_positions if position not in max_forced]
    return {
        "runs": runs,
        "lengths": lengths,
        "complete_lengths": complete_lengths,
        "boundary_positions": boundary_positions,
        "max_forced_positions": max_forced_positions,
        "learned_positions": learned_positions,
    }


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if scores.size == 0 or labels.size != scores.size:
        return None
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def pearson(values_a: np.ndarray, values_b: np.ndarray) -> Optional[float]:
    if values_a.size < 2 or values_a.size != values_b.size:
        return None
    std_a = float(values_a.std())
    std_b = float(values_b.std())
    if std_a == 0.0 or std_b == 0.0:
        return None
    return float(np.corrcoef(values_a, values_b)[0, 1])


def histogram_quantile(histogram: Mapping[int, int], quantile: float) -> Optional[float]:
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return None
    target_rank = max(1, int(math.ceil(quantile * total)))
    cumulative = 0
    for length, count in sorted((int(length), int(count)) for length, count in histogram.items()):
        cumulative += count
        if cumulative >= target_rank:
            return float(length)
    return float(max(histogram))


def concatenate(values: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [value.reshape(-1) for value in values if value.size > 0]
    if not nonempty:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(nonempty).astype(np.float64, copy=False)


def update_pos_metrics(stats: DocumentStats, runs, syntax_labels: np.ndarray) -> None:
    if syntax_labels.size < 2:
        return
    for _, start, end in runs:
        labels = syntax_labels[start:end]
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        counts = np.bincount(labels.astype(np.int64))
        probabilities = counts[counts > 0].astype(np.float64) / labels.size
        stats.segment_pos_purity_sum += float(probabilities.max())
        stats.segment_pos_entropy_sum += float(-(probabilities * np.log(probabilities)).sum())
        stats.segment_pos_count += 1


def update_row_statistics(
    accumulator: DatasetAccumulator,
    window: AnalysisWindow,
    segment_ids: torch.Tensor,
    boundary_probs: Optional[torch.Tensor],
    gate_values: Optional[torch.Tensor],
    max_chunk_len: int,
    tokenizer,
    decode_metrics: bool,
    num_examples: int,
) -> None:
    stats = accumulator.documents.setdefault(window.source_index, DocumentStats(source_index=window.source_index))
    valid_length = int(window.tokens.numel())
    segments = segment_ids[:valid_length].detach().cpu().long().tolist()
    classification = classify_segmentation(
        segments,
        max_chunk_len=max_chunk_len,
        left_censored=window.left_censored,
        right_censored=window.right_censored,
    )
    runs = classification["runs"]
    boundary_positions = classification["boundary_positions"]
    max_forced_positions = classification["max_forced_positions"]
    learned_positions = classification["learned_positions"]
    lengths = classification["lengths"]
    complete_lengths = classification["complete_lengths"]

    stats.windows += 1
    stats.tokens += valid_length
    stats.concepts += len(runs)
    stats.complete_segments += len(complete_lengths)
    stats.length_histogram.update(lengths)
    stats.complete_length_histogram.update(complete_lengths)
    stats.boundary_eligible += max(0, valid_length - 1)
    stats.observed_boundaries += len(boundary_positions)
    stats.max_forced_boundaries += len(max_forced_positions)
    stats.learned_boundaries += len(learned_positions)
    if window.word_counts is not None:
        stats.source_words += float(window.word_counts.sum().item())

    hard_labels = np.zeros(max(0, valid_length - 1), dtype=np.int8)
    for position in boundary_positions:
        if position > 0:
            hard_labels[position - 1] = 1

    probabilities = None
    if boundary_probs is not None and valid_length > 1:
        probabilities = boundary_probs[1:valid_length].detach().float().cpu().numpy()
        accumulator.boundary_probabilities.append(probabilities)
        accumulator.hard_boundary_labels.append(hard_labels)
        stats.boundary_probability_sum += float(probabilities.sum())
        stats.boundary_probability_count += int(probabilities.size)
        boundary_mask = hard_labels.astype(bool)
        stats.hard_boundary_probability_sum += float(probabilities[boundary_mask].sum())
        stats.hard_boundary_probability_count += int(boundary_mask.sum())
        stats.interior_boundary_probability_sum += float(probabilities[~boundary_mask].sum())
        stats.interior_boundary_probability_count += int((~boundary_mask).sum())

    gates = None
    if gate_values is not None:
        gates = gate_values[:valid_length].detach().float().cpu().numpy()
        accumulator.gate_values.append(gates)
        stats.gate_sum += float(gates.sum())
        stats.gate_square_sum += float(np.square(gates).sum())
        stats.gate_count += int(gates.size)
        if valid_length > 1:
            transition_gates = gates[1:]
            accumulator.gate_transition_values.append(transition_gates)
            boundary_mask = hard_labels.astype(bool)
            stats.boundary_gate_sum += float(transition_gates[boundary_mask].sum())
            stats.boundary_gate_count += int(boundary_mask.sum())
            stats.interior_gate_sum += float(transition_gates[~boundary_mask].sum())
            stats.interior_gate_count += int((~boundary_mask).sum())
            if probabilities is not None:
                accumulator.gate_boundary_probabilities.append(probabilities)

    if window.word_boundaries is not None and valid_length > 1:
        labels = window.word_boundaries[1:valid_length].detach().cpu().numpy()
        valid = labels >= 0
        gold = labels[valid] == 1
        predicted = hard_labels[valid].astype(bool)
        stats.word_tp += int(np.logical_and(predicted, gold).sum())
        stats.word_fp += int(np.logical_and(predicted, ~gold).sum())
        stats.word_fn += int(np.logical_and(~predicted, gold).sum())
        stats.word_tn += int(np.logical_and(~predicted, ~gold).sum())
        stats.word_start_positions += int(gold.sum())
        stats.word_inside_positions += int((~gold).sum())
        stats.boundaries_at_word_starts += int(predicted[gold].sum())
        stats.boundaries_inside_words += int(predicted[~gold].sum())
        if probabilities is not None:
            accumulator.word_boundary_probabilities.append(probabilities[valid])
            accumulator.word_boundary_labels.append(gold.astype(np.int8))

    if window.syntax_labels is not None and valid_length > 1:
        syntax = window.syntax_labels[:valid_length].detach().cpu().numpy()
        valid = (syntax[1:] >= 0) & (syntax[:-1] >= 0)
        changes = syntax[1:] != syntax[:-1]
        predicted = hard_labels.astype(bool)
        stats.syntax_change_positions += int(np.logical_and(valid, changes).sum())
        stats.syntax_same_positions += int(np.logical_and(valid, ~changes).sum())
        stats.boundaries_at_syntax_changes += int(np.logical_and.reduce((valid, changes, predicted)).sum())
        stats.boundaries_at_syntax_same += int(np.logical_and.reduce((valid, ~changes, predicted)).sum())
        update_pos_metrics(stats, runs, syntax)

    decoded_segments: Optional[List[str]] = None
    if decode_metrics or len(accumulator.examples) < num_examples:
        token_slices = [window.tokens[start:end].tolist() for _, start, end in runs]
        decoded_segments = tokenizer.batch_decode(
            token_slices,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    if decode_metrics and decoded_segments is not None:
        usable = list(range(len(decoded_segments)))
        if window.left_censored and usable:
            usable = usable[1:]
        if window.right_censored and usable:
            usable = usable[:-1]
        for index in usable:
            decoded = decoded_segments[index]
            stats.decoded_nonspace_characters += sum(1 for character in decoded if not character.isspace())
            stats.decoded_whitespace_units += len(decoded.split())
            stats.decoded_segment_count += 1

    if len(accumulator.examples) < num_examples and decoded_segments is not None:
        example = {
            "source_document_index": window.source_index,
            "left_censored": window.left_censored,
            "right_censored": window.right_censored,
            "text": tokenizer.decode(
                window.tokens.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "segments": decoded_segments,
            "segment_lengths": lengths,
            "boundary_probabilities_at_segment_starts": [
                float(boundary_probs[position].item()) for position in boundary_positions
            ]
            if boundary_probs is not None
            else None,
            "gate_means_by_segment": [
                float(gates[start:end].mean()) for _, start, end in runs
            ]
            if gates is not None
            else None,
        }
        accumulator.examples.append(example)


def aggregate_document_stats(documents: Sequence[DocumentStats], max_chunk_len: int) -> Dict[str, object]:
    combined = DocumentStats(source_index=-1)
    for stats in documents:
        for field_name in (
            "windows",
            "tokens",
            "concepts",
            "complete_segments",
            "boundary_eligible",
            "observed_boundaries",
            "learned_boundaries",
            "max_forced_boundaries",
            "boundary_probability_count",
            "hard_boundary_probability_count",
            "interior_boundary_probability_count",
            "gate_count",
            "boundary_gate_count",
            "interior_gate_count",
            "word_tp",
            "word_fp",
            "word_fn",
            "word_tn",
            "word_start_positions",
            "word_inside_positions",
            "boundaries_at_word_starts",
            "boundaries_inside_words",
            "syntax_change_positions",
            "syntax_same_positions",
            "boundaries_at_syntax_changes",
            "boundaries_at_syntax_same",
            "segment_pos_count",
            "decoded_nonspace_characters",
            "decoded_whitespace_units",
            "decoded_segment_count",
        ):
            setattr(combined, field_name, getattr(combined, field_name) + getattr(stats, field_name))
        for field_name in (
            "boundary_probability_sum",
            "hard_boundary_probability_sum",
            "interior_boundary_probability_sum",
            "gate_sum",
            "gate_square_sum",
            "boundary_gate_sum",
            "interior_gate_sum",
            "segment_pos_purity_sum",
            "segment_pos_entropy_sum",
            "source_words",
        ):
            setattr(combined, field_name, getattr(combined, field_name) + getattr(stats, field_name))
        combined.length_histogram.update(stats.length_histogram)
        combined.complete_length_histogram.update(stats.complete_length_histogram)

    primary_histogram = combined.complete_length_histogram or combined.length_histogram
    segment_total = sum(primary_histogram.values())
    length_sum = sum(int(length) * int(count) for length, count in primary_histogram.items())
    word_metrics = binary_metrics(combined.word_tp, combined.word_fp, combined.word_fn, combined.word_tn)
    gate_mean = safe_ratio(combined.gate_sum, combined.gate_count)
    gate_variance = None
    if gate_mean is not None:
        gate_variance = max(0.0, combined.gate_square_sum / combined.gate_count - gate_mean * gate_mean)
    metrics: Dict[str, object] = {
        "documents": len(documents),
        "windows": combined.windows,
        "tokens": combined.tokens,
        "concepts": combined.concepts,
        "tokens_per_concept": safe_ratio(combined.tokens, combined.concepts),
        "source_words": combined.source_words if combined.source_words > 0 else None,
        "source_words_per_concept": safe_ratio(combined.source_words, combined.concepts)
        if combined.source_words > 0
        else None,
        "complete_segments": combined.complete_segments,
        "mean_segment_length": safe_ratio(length_sum, segment_total),
        "p50_segment_length": histogram_quantile(primary_histogram, 0.50),
        "p90_segment_length": histogram_quantile(primary_histogram, 0.90),
        "single_token_segment_fraction": safe_ratio(primary_histogram.get(1, 0), segment_total),
        "max_length_segment_fraction": safe_ratio(primary_histogram.get(max_chunk_len, 0), segment_total),
        "boundary_eligible_transitions": combined.boundary_eligible,
        "observed_boundary_rate": safe_ratio(combined.observed_boundaries, combined.boundary_eligible),
        "learned_boundary_rate": safe_ratio(combined.learned_boundaries, combined.boundary_eligible),
        "max_forced_boundary_rate": safe_ratio(combined.max_forced_boundaries, combined.boundary_eligible),
        "boundary_probability_mean": safe_ratio(
            combined.boundary_probability_sum,
            combined.boundary_probability_count,
        ),
        "boundary_probability_at_hard_boundaries": safe_ratio(
            combined.hard_boundary_probability_sum,
            combined.hard_boundary_probability_count,
        ),
        "boundary_probability_at_interiors": safe_ratio(
            combined.interior_boundary_probability_sum,
            combined.interior_boundary_probability_count,
        ),
        "lm_residual_gate_mean": gate_mean,
        "lm_residual_gate_std": math.sqrt(gate_variance) if gate_variance is not None else None,
        "lm_residual_gate_at_boundaries": safe_ratio(combined.boundary_gate_sum, combined.boundary_gate_count),
        "lm_residual_gate_at_interiors": safe_ratio(combined.interior_gate_sum, combined.interior_gate_count),
        "word_boundary_precision": word_metrics["precision"],
        "word_boundary_recall": word_metrics["recall"],
        "word_boundary_f1": word_metrics["f1"],
        "word_boundary_accuracy": word_metrics["accuracy"],
        "boundary_rate_at_word_starts": safe_ratio(
            combined.boundaries_at_word_starts,
            combined.word_start_positions,
        ),
        "boundary_rate_inside_words": safe_ratio(
            combined.boundaries_inside_words,
            combined.word_inside_positions,
        ),
        "boundary_rate_at_syntax_changes": safe_ratio(
            combined.boundaries_at_syntax_changes,
            combined.syntax_change_positions,
        ),
        "boundary_rate_without_syntax_change": safe_ratio(
            combined.boundaries_at_syntax_same,
            combined.syntax_same_positions,
        ),
        "segment_pos_purity": safe_ratio(combined.segment_pos_purity_sum, combined.segment_pos_count),
        "segment_pos_entropy": safe_ratio(combined.segment_pos_entropy_sum, combined.segment_pos_count),
        "decoded_nonspace_characters_per_concept": safe_ratio(
            combined.decoded_nonspace_characters,
            combined.decoded_segment_count,
        ),
        "decoded_whitespace_units_per_concept": safe_ratio(
            combined.decoded_whitespace_units,
            combined.decoded_segment_count,
        ),
    }
    word_start_rate = metrics["boundary_rate_at_word_starts"]
    word_inside_rate = metrics["boundary_rate_inside_words"]
    metrics["word_boundary_rate_delta"] = (
        word_start_rate - word_inside_rate
        if word_start_rate is not None and word_inside_rate is not None
        else None
    )
    syntax_change_rate = metrics["boundary_rate_at_syntax_changes"]
    syntax_same_rate = metrics["boundary_rate_without_syntax_change"]
    metrics["syntax_change_boundary_delta"] = (
        syntax_change_rate - syntax_same_rate
        if syntax_change_rate is not None and syntax_same_rate is not None
        else None
    )
    return {
        "metrics": metrics,
        "length_histogram": {str(length): int(primary_histogram.get(length, 0)) for length in range(1, max_chunk_len + 1)},
        "all_length_histogram": {
            str(length): int(count) for length, count in sorted(combined.length_histogram.items())
        },
        "counts": {
            "observed_boundaries": combined.observed_boundaries,
            "learned_boundaries": combined.learned_boundaries,
            "max_forced_boundaries": combined.max_forced_boundaries,
            "word_tp": combined.word_tp,
            "word_fp": combined.word_fp,
            "word_fn": combined.word_fn,
            "word_tn": combined.word_tn,
        },
    }


def add_distribution_metrics(result: Dict[str, object], accumulator: DatasetAccumulator) -> None:
    metrics = result["metrics"]
    boundary_probabilities = concatenate(accumulator.boundary_probabilities)
    hard_labels = concatenate(accumulator.hard_boundary_labels).astype(np.int8)
    metrics["boundary_probability_hard_boundary_auc"] = auc_rank(boundary_probabilities, hard_labels)

    gate_values = concatenate(accumulator.gate_values)
    gate_transition_values = concatenate(accumulator.gate_transition_values)
    gate_boundary_probabilities = concatenate(accumulator.gate_boundary_probabilities)
    if gate_values.size > 0:
        metrics["lm_residual_gate_p10"] = float(np.quantile(gate_values, 0.10))
        metrics["lm_residual_gate_p50"] = float(np.quantile(gate_values, 0.50))
        metrics["lm_residual_gate_p90"] = float(np.quantile(gate_values, 0.90))
        if gate_transition_values.size == gate_boundary_probabilities.size:
            metrics["lm_residual_gate_boundary_probability_correlation"] = pearson(
                gate_transition_values,
                gate_boundary_probabilities,
            )
        counts, edges = np.histogram(gate_values, bins=40)
        result["gate_histogram"] = {
            "edges": [float(value) for value in edges.tolist()],
            "counts": [int(value) for value in counts.tolist()],
        }
    else:
        metrics["lm_residual_gate_p10"] = None
        metrics["lm_residual_gate_p50"] = None
        metrics["lm_residual_gate_p90"] = None
        metrics["lm_residual_gate_boundary_probability_correlation"] = None
        result["gate_histogram"] = None

    word_probabilities = concatenate(accumulator.word_boundary_probabilities)
    word_labels = concatenate(accumulator.word_boundary_labels).astype(np.int8)
    metrics["word_boundary_probability_auc"] = auc_rank(word_probabilities, word_labels)


def bootstrap_metrics(
    documents: Sequence[DocumentStats],
    max_chunk_len: int,
    samples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if samples <= 0 or len(documents) < 2:
        return {}
    rng = np.random.default_rng(seed)
    tracked = (
        "tokens_per_concept",
        "source_words_per_concept",
        "mean_segment_length",
        "p50_segment_length",
        "p90_segment_length",
        "single_token_segment_fraction",
        "max_length_segment_fraction",
        "observed_boundary_rate",
        "learned_boundary_rate",
        "max_forced_boundary_rate",
        "lm_residual_gate_mean",
        "word_boundary_precision",
        "word_boundary_recall",
        "word_boundary_f1",
        "word_boundary_rate_delta",
        "syntax_change_boundary_delta",
        "decoded_nonspace_characters_per_concept",
    )
    distributions: Dict[str, List[float]] = {name: [] for name in tracked}
    for _ in range(samples):
        indices = rng.integers(0, len(documents), size=len(documents))
        resampled = [documents[int(index)] for index in indices]
        metrics = aggregate_document_stats(resampled, max_chunk_len)["metrics"]
        for name in tracked:
            value = metrics.get(name)
            if value is not None and math.isfinite(float(value)):
                distributions[name].append(float(value))
    alpha = 1.0 - confidence
    intervals: Dict[str, Dict[str, float]] = {}
    for name, values in distributions.items():
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "lower": float(np.quantile(array, alpha / 2.0)),
            "upper": float(np.quantile(array, 1.0 - alpha / 2.0)),
            "confidence": float(confidence),
            "samples": int(samples),
        }
    return intervals


def analyze_dataset(
    model,
    tokenizer,
    spec: DatasetSpec,
    documents: Sequence[SampledDocument],
    scanned_documents: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[DocumentStats]]:
    separator_ids = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    windows = build_windows(
        documents,
        spec,
        seq_length=args.seq_length,
        windows_per_document=args.windows_per_document,
        seed=args.seed,
        separator_ids=separator_ids,
    )
    if not windows:
        raise RuntimeError(f"No analyzable windows were produced for {spec.name}.")

    base_model = getattr(model, "concept_gpt_bert", None)
    if base_model is None:
        raise TypeError(
            f"{model.__class__.__name__} does not expose concept_gpt_bert; this analyzer targets BabyConceptLM checkpoints."
        )
    max_chunk_len = int(getattr(model.config, "max_chunk_len", 8))
    accumulator = DatasetAccumulator()

    with torch.no_grad():
        for start in range(0, len(windows), args.batch_size):
            batch_windows = windows[start : start + args.batch_size]
            batch = collate_windows(batch_windows, tokenizer.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            language_ids = batch["language_ids"].to(device)
            syntax_token_labels = batch["syntax_token_labels"]
            if syntax_token_labels is not None:
                syntax_token_labels = syntax_token_labels.to(device)
            with autocast_context(device, args.dtype):
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    language_ids=language_ids,
                    syntax_token_labels=syntax_token_labels,
                    mode="clm",
                    return_dict=True,
                )
                gate_values = compute_lm_residual_gate(model, outputs, attention_mask)
            if outputs.segment_ids is None:
                raise RuntimeError("Model output lacks segment_ids.")
            for row, window in enumerate(batch_windows):
                update_row_statistics(
                    accumulator=accumulator,
                    window=window,
                    segment_ids=outputs.segment_ids[row],
                    boundary_probs=outputs.boundary_probs[row] if outputs.boundary_probs is not None else None,
                    gate_values=gate_values[row] if gate_values is not None else None,
                    max_chunk_len=max_chunk_len,
                    tokenizer=tokenizer,
                    decode_metrics=not args.skip_decode_metrics,
                    num_examples=args.num_examples,
                )

    document_stats = list(accumulator.documents.values())
    result = aggregate_document_stats(document_stats, max_chunk_len)
    add_distribution_metrics(result, accumulator)
    result["confidence_intervals"] = bootstrap_metrics(
        document_stats,
        max_chunk_len=max_chunk_len,
        samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=stable_seed(args.seed, spec.name),
    )
    result["dataset"] = {
        "name": spec.name,
        "path": str(spec.path),
        "language_id": spec.language_id,
        "word_boundary_path": str(spec.word_boundary_path) if spec.word_boundary_path else None,
        "syntax_path": str(spec.syntax_path) if spec.syntax_path else None,
        "word_count_path": str(spec.word_count_path) if spec.word_count_path else None,
        "documents_scanned": scanned_documents,
        "documents_selected": len(documents),
        "windows_analyzed": len(windows),
        "data_fingerprint": file_fingerprint(spec.path),
        "word_boundary_fingerprint": file_fingerprint(spec.word_boundary_path),
        "syntax_fingerprint": file_fingerprint(spec.syntax_path),
        "word_count_fingerprint": file_fingerprint(spec.word_count_path),
    }
    result["examples"] = accumulator.examples
    return result, document_stats


@lru_cache(maxsize=None)
def file_fingerprint(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def model_metadata(model_spec: ModelSpec, model, tokenizer_path: Path) -> Dict[str, object]:
    return {
        "name": model_spec.name,
        "path": str(model_spec.path),
        "class": model.__class__.__name__,
        "model_variant": str(getattr(model.config, "model_variant", "")),
        "num_token_layers": int(getattr(model.config, "num_token_layers", -1)),
        "num_concept_layers": int(getattr(model.config, "num_concept_layers", -1)),
        "num_readout_layers": int(getattr(model.config, "num_readout_layers", -1)),
        "max_chunk_len": int(getattr(model.config, "max_chunk_len", -1)),
        "lm_concept_residual_gate": bool(getattr(model.config, "lm_concept_residual_gate", False)),
        "lm_concept_gate_max": float(getattr(model.config, "lm_concept_gate_max", 0.0)),
        "config_fingerprint": file_fingerprint(model_spec.path / "config.json"),
        "tokenizer_fingerprint": file_fingerprint(
            tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
        ),
        "train_args_summary": json.loads((model_spec.path / "train_args_summary.json").read_text(encoding="utf-8"))
        if (model_spec.path / "train_args_summary.json").is_file()
        else None,
    }


def flatten_summary_row(model_name: str, dataset_result: Mapping[str, object]) -> Dict[str, object]:
    metrics = dataset_result["metrics"]
    row: Dict[str, object] = {"model": model_name, "dataset": dataset_result["dataset"]["name"]}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) or value is None:
            row[key] = value
    intervals = dataset_result.get("confidence_intervals", {})
    for metric_name, interval in intervals.items():
        row[f"{metric_name}_ci_lower"] = interval["lower"]
        row[f"{metric_name}_ci_upper"] = interval["upper"]
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def paper_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Model | Language | Tokens/concept | Words/concept | Learned boundary % | P50/P90 | P(L=8) % | Gate mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        learned = row.get("learned_boundary_rate")
        max_fraction = row.get("max_length_segment_fraction")
        lines.append(
            "| {model} | {dataset} | {tokens} | {words} | {learned} | {p50}/{p90} | {max_fraction} | {gate} |".format(
                model=row["model"],
                dataset=row["dataset"],
                tokens=format_number(row.get("tokens_per_concept")),
                words=format_number(row.get("source_words_per_concept")),
                learned=format_number(100.0 * learned if learned is not None else None, 2),
                p50=format_number(row.get("p50_segment_length"), 1),
                p90=format_number(row.get("p90_segment_length"), 1),
                max_fraction=format_number(100.0 * max_fraction if max_fraction is not None else None, 2),
                gate=format_number(row.get("lm_residual_gate_mean"), 4),
            )
        )
    lines.append("")
    lines.append(
        "Primary length statistics exclude segments censored by sampled window edges. Learned boundary rate excludes cuts attributed to the hard maximum length."
    )
    return "\n".join(lines) + "\n"


def write_document_jsonl(path: Path, records: Sequence[Tuple[str, str, DocumentStats]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for model_name, dataset_name, stats in records:
            payload = asdict(stats)
            payload["model"] = model_name
            payload["dataset"] = dataset_name
            payload["length_histogram"] = dict(stats.length_histogram)
            payload["complete_length_histogram"] = dict(stats.complete_length_histogram)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_plots(summary: Mapping[str, object], output_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plots.", file=sys.stderr)
        return []

    series = []
    for model_result in summary["models"]:
        for dataset_result in model_result["datasets"]:
            series.append((model_result["model"]["name"], dataset_result))
    if not series:
        return []

    plot_paths: List[Path] = []
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for model_name, dataset_result in series:
        histogram = dataset_result["length_histogram"]
        lengths = np.asarray([int(value) for value in histogram], dtype=np.int64)
        counts = np.asarray([histogram[str(length)] for length in lengths], dtype=np.float64)
        fractions = counts / counts.sum() if counts.sum() else counts
        axis.plot(lengths, fractions, marker="o", linewidth=1.5, label=f"{model_name}/{dataset_result['dataset']['name']}")
    axis.set_xlabel("Segment length (model tokens)")
    axis.set_ylabel("Fraction of complete segments")
    axis.set_xticks(sorted({int(length) for _, result in series for length in result["length_histogram"]}))
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    length_path = output_dir / "segment_length_distribution.png"
    figure.savefig(length_path, dpi=220)
    plt.close(figure)
    plot_paths.append(length_path)

    labels = [f"{model}/{result['dataset']['name']}" for model, result in series]
    token_values = [result["metrics"]["tokens_per_concept"] for _, result in series]
    boundary_values = [100.0 * result["metrics"]["learned_boundary_rate"] for _, result in series]
    max_values = [100.0 * result["metrics"]["max_length_segment_fraction"] for _, result in series]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, max(3.2, 0.32 * len(labels) + 1.4)))
    y = np.arange(len(labels))
    for axis, values, title in zip(
        axes,
        (token_values, boundary_values, max_values),
        ("Tokens / concept", "Learned boundaries (%)", "Segments at max length (%)"),
    ):
        axis.barh(y, values)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, labels=labels, fontsize=7)
    for axis in axes[1:]:
        axis.set_yticks(y, labels=[])
    figure.tight_layout()
    summary_path = output_dir / "segmentation_summary.png"
    figure.savefig(summary_path, dpi=220)
    plt.close(figure)
    plot_paths.append(summary_path)
    return plot_paths


def log_to_wandb(
    args: argparse.Namespace,
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    plot_paths: Sequence[Path],
) -> None:
    if args.report_to != "wandb":
        return
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install wandb or run with --report_to none.") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=summary["analysis_config"],
    )
    scalar_metrics: Dict[str, float] = {}
    for row in rows:
        prefix = f"segmentation/{row['model']}/{row['dataset']}"
        for key, value in row.items():
            if key in {"model", "dataset"} or not isinstance(value, (int, float)):
                continue
            scalar_metrics[f"{prefix}/{key}"] = float(value)
    table = wandb.Table(columns=list(rows[0].keys()) if rows else ["model", "dataset"])
    for row in rows:
        table.add_data(*(row.get(column) for column in table.columns))
    payload: Dict[str, object] = {**scalar_metrics, "segmentation/summary": table}
    for plot_path in plot_paths:
        payload[f"segmentation/plots/{plot_path.stem}"] = wandb.Image(str(plot_path))
    run.log(payload)
    run.finish()


def main() -> None:
    args = parse_args()
    if not args.accept_remote_code:
        raise ValueError("Review checkpoint code and pass --accept_remote_code to enable inference.")
    if args.seq_length <= 1:
        raise ValueError("--seq_length must be greater than 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.windows_per_document < 0:
        raise ValueError("--windows_per_document must be non-negative; zero means all non-overlapping windows.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between zero and one.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_specs = collect_model_specs(args)
    dataset_specs = collect_dataset_specs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modules_cache = (
        Path(args.hf_modules_cache).expanduser().resolve()
        if args.hf_modules_cache
        else Path(os.environ.get("HF_MODULES_CACHE", output_dir / ".hf_modules")).expanduser().resolve()
    )
    modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)
    device = resolve_device(args.device)

    sampled_datasets: Dict[str, Tuple[List[SampledDocument], int]] = {}
    for dataset_spec in dataset_specs:
        sampled_datasets[dataset_spec.name] = sample_documents(
            dataset_spec,
            max_documents=args.max_documents,
            sampling=args.document_sampling,
            seed=args.seed,
        )

    from transformers import AutoModelForCausalLM

    all_model_results: List[Dict[str, object]] = []
    document_records: List[Tuple[str, str, DocumentStats]] = []
    summary_rows: List[Dict[str, object]] = []
    for model_spec in model_specs:
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else model_spec.path
        tokenizer = load_tokenizer(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_spec.path),
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        model.eval()
        if args.seq_length > int(getattr(model.config, "max_position_embeddings", args.seq_length)):
            raise ValueError(
                f"--seq_length={args.seq_length} exceeds max_position_embeddings="
                f"{model.config.max_position_embeddings} for {model_spec.name}."
            )

        dataset_results = []
        for dataset_spec in dataset_specs:
            documents, scanned_documents = sampled_datasets[dataset_spec.name]
            dataset_result, per_document = analyze_dataset(
                model=model,
                tokenizer=tokenizer,
                spec=dataset_spec,
                documents=documents,
                scanned_documents=scanned_documents,
                args=args,
                device=device,
            )
            dataset_results.append(dataset_result)
            summary_rows.append(flatten_summary_row(model_spec.name, dataset_result))
            document_records.extend(
                (model_spec.name, dataset_spec.name, document_stats) for document_stats in per_document
            )

        all_model_results.append(
            {
                "model": model_metadata(model_spec, model, tokenizer_path),
                "datasets": dataset_results,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: Dict[str, object] = {
        "analysis_format": "babyconceptlm_segmentation_analysis_v1",
        "analysis_config": {
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "max_documents": args.max_documents,
            "document_sampling": args.document_sampling,
            "windows_per_document": args.windows_per_document,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": args.confidence,
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "decode_metrics": not args.skip_decode_metrics,
            "hf_modules_cache": str(modules_cache),
            "manifest": str(Path(args.manifest).expanduser().resolve()) if args.manifest else None,
            "manifest_fingerprint": file_fingerprint(Path(args.manifest).expanduser().resolve())
            if args.manifest
            else None,
        },
        "definitions": {
            "tokens_per_concept": "Total analyzed non-special model tokens divided by predicted concepts.",
            "learned_boundary_rate": "Predicted segment starts excluding window starts and cuts attributed to max_chunk_len.",
            "max_forced_boundary_rate": "Segment starts whose preceding segment reached max_chunk_len.",
            "length_statistics": "Computed over segments not censored by sampled window edges.",
            "source_words_per_concept": "Token-aligned source word counts divided by predicted concepts when a sidecar is available.",
            "confidence_intervals": "Percentile intervals from document-level bootstrap resampling.",
        },
        "models": all_model_results,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "summary.csv", summary_rows)

    distribution_rows = []
    for model_result in all_model_results:
        model_name = model_result["model"]["name"]
        for dataset_result in model_result["datasets"]:
            histogram = dataset_result["length_histogram"]
            total = sum(histogram.values())
            for length, count in histogram.items():
                distribution_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_result["dataset"]["name"],
                        "length": int(length),
                        "count": int(count),
                        "fraction": float(count) / total if total else None,
                    }
                )
    write_csv(output_dir / "segment_length_distribution.csv", distribution_rows)
    write_document_jsonl(output_dir / "document_metrics.jsonl", document_records)
    (output_dir / "paper_table.md").write_text(paper_table(summary_rows), encoding="utf-8")
    plot_paths = [] if args.no_plots else create_plots(summary, output_dir)
    log_to_wandb(args, summary, summary_rows, plot_paths)

    print(f"Wrote segmentation analysis to {output_dir}")
    print(f"Summary: {summary_path}")
    print((output_dir / "paper_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

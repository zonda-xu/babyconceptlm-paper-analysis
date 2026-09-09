from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers


REPO_ROOT = Path(__file__).resolve().parents[1]
NONSPACE_RE = re.compile(r"\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
ALLOWED_MEMORY_TYPES = {"recalled", "imagined", "retold"}


@dataclass(frozen=True)
class StoryRecord:
    source_row: int
    assignment_id: str
    memory_type: str
    text: str
    text_sha256: str
    word_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and tokenize Hippocorpus for BabyConceptLM STRICT segmentation analysis."
    )
    parser.add_argument(
        "--source_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
    )
    parser.add_argument("--min_words", type=int, default=100)
    parser.add_argument("--max_words", type=int, default=800)
    parser.add_argument("--allow_urls", action="store_true")
    parser.add_argument("--allow_emails", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clean_story(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_stories(args: argparse.Namespace) -> tuple[list[StoryRecord], Counter[str], Counter[str]]:
    records: list[StoryRecord] = []
    rejected: Counter[str] = Counter()
    memory_types: Counter[str] = Counter()
    seen_texts: set[str] = set()
    seen_assignment_ids: set[str] = set()
    with args.source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"AssignmentId", "memType", "story"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing Hippocorpus columns: {sorted(missing_columns)}")
        for source_row, row in enumerate(reader, start=2):
            assignment_id = str(row.get("AssignmentId") or "").strip()
            memory_type = str(row.get("memType") or "").strip().lower()
            text = clean_story(str(row.get("story") or ""))
            if not assignment_id:
                rejected["missing_assignment_id"] += 1
                continue
            if assignment_id in seen_assignment_ids:
                rejected["duplicate_assignment_id"] += 1
                continue
            if memory_type not in ALLOWED_MEMORY_TYPES:
                rejected["unknown_memory_type"] += 1
                continue
            if not text:
                rejected["empty_story"] += 1
                continue
            if not args.allow_urls and URL_RE.search(text):
                rejected["contains_url"] += 1
                continue
            if not args.allow_emails and EMAIL_RE.search(text):
                rejected["contains_email"] += 1
                continue
            word_count = len(NONSPACE_RE.findall(text))
            if word_count < args.min_words:
                rejected["below_min_words"] += 1
                continue
            if args.max_words > 0 and word_count > args.max_words:
                rejected["above_max_words"] += 1
                continue
            text_sha256 = sha256_bytes(text.casefold().encode("utf-8"))
            if text_sha256 in seen_texts:
                rejected["duplicate_normalized_story"] += 1
                continue
            seen_assignment_ids.add(assignment_id)
            seen_texts.add(text_sha256)
            memory_types[memory_type] += 1
            records.append(
                StoryRecord(
                    source_row=source_row,
                    assignment_id=assignment_id,
                    memory_type=memory_type,
                    text=text,
                    text_sha256=text_sha256,
                    word_count=word_count,
                )
            )
    if not records:
        raise ValueError("No Hippocorpus stories survived preprocessing.")
    return records, rejected, memory_types


def parse_merges(raw_merges: Iterable[object]) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for raw_merge in raw_merges:
        if isinstance(raw_merge, str):
            pieces = raw_merge.split(" ", 1)
        elif isinstance(raw_merge, (list, tuple)):
            pieces = [str(piece) for piece in raw_merge]
        else:
            raise TypeError(f"Unsupported BPE merge: {raw_merge!r}")
        if len(pieces) != 2:
            raise ValueError(f"Expected a two-piece BPE merge, got {raw_merge!r}")
        merges.append((pieces[0], pieces[1]))
    return merges


def load_submission_tokenizer(path: Path) -> tuple[Tokenizer, Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_payload = payload.get("model") or {}
    if model_payload.get("type") != "BPE":
        raise ValueError(f"Expected a BPE tokenizer, got {model_payload.get('type')!r}")
    model_kwargs: dict[str, object] = {
        "vocab": model_payload["vocab"],
        "merges": parse_merges(model_payload.get("merges", [])),
        "fuse_unk": bool(model_payload.get("fuse_unk", False)),
        "byte_fallback": bool(model_payload.get("byte_fallback", False)),
        "ignore_merges": bool(model_payload.get("ignore_merges", False)),
    }
    for optional_name in ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"):
        optional_value = model_payload.get(optional_name)
        if optional_value is not None:
            model_kwargs[optional_name] = optional_value
    tokenizer = Tokenizer(models.BPE(**model_kwargs))
    normalizer_payload = payload.get("normalizer") or {}
    normalizer_types = [
        item.get("type")
        for item in normalizer_payload.get("normalizers", [])
        if isinstance(item, Mapping)
    ]
    if normalizer_payload.get("type") == "NFKC" or normalizer_types == ["NFKC"]:
        tokenizer.normalizer = normalizers.NFKC()
    else:
        raise ValueError(f"Unsupported tokenizer normalizer: {normalizer_payload!r}")
    pre_tokenizer_payload = payload.get("pre_tokenizer") or {}
    if pre_tokenizer_payload.get("type") != "ByteLevel":
        raise ValueError(f"Unsupported pre-tokenizer: {pre_tokenizer_payload!r}")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=bool(pre_tokenizer_payload.get("add_prefix_space", False)),
        use_regex=bool(pre_tokenizer_payload.get("use_regex", True)),
    )
    decoder_payload = payload.get("decoder") or {}
    if decoder_payload.get("type") != "ByteLevel":
        raise ValueError(f"Unsupported tokenizer decoder: {decoder_payload!r}")
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer, payload


def whitespace_word_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in NONSPACE_RE.finditer(text)]


def token_aligned_word_counts(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    counts = [0] * len(offsets)
    token_index = 0
    for word_start, word_end in whitespace_word_spans(text):
        while token_index < len(offsets) and offsets[token_index][1] <= word_start:
            token_index += 1
        matched_index = token_index
        while matched_index < len(offsets):
            token_start, token_end = offsets[matched_index]
            if token_start >= word_end:
                break
            if token_end > word_start and token_start < word_end:
                counts[matched_index] += 1
                break
            matched_index += 1
        if matched_index >= len(offsets) or offsets[matched_index][0] >= word_end:
            raise ValueError(f"Could not align word span {(word_start, word_end)} in {text!r}")
    return counts


def atomic_torch_save(value: object, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def prepare(args: argparse.Namespace) -> Mapping[str, object]:
    source_csv = args.source_csv.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output_prefix = args.output_prefix.resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if args.min_words < 1:
        raise ValueError("--min_words must be positive.")
    if args.max_words > 0 and args.max_words < args.min_words:
        raise ValueError("--max_words must be zero or at least --min_words.")

    paths = {
        "tokenized_bin": output_prefix.with_name(f"{output_prefix.name}_tokenized.bin"),
        "word_counts": output_prefix.with_name(f"{output_prefix.name}_word_counts.pt"),
        "word_boundaries": output_prefix.with_name(f"{output_prefix.name}_word_boundaries.pt"),
        "index": output_prefix.with_name(f"{output_prefix.name}_index.jsonl"),
        "manifest": output_prefix.with_name(f"{output_prefix.name}.meta.json"),
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output files already exist; pass --overwrite: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    records, rejected, memory_types = read_stories(args)
    tokenizer, tokenizer_payload = load_submission_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    token_dtype = torch.int16 if vocab_size <= 32767 else torch.int32
    token_documents: list[torch.Tensor] = []
    word_count_documents: list[torch.Tensor] = []
    word_boundary_documents: list[torch.Tensor] = []
    index_rows: list[Mapping[str, object]] = []
    token_lengths: list[int] = []
    word_lengths: list[int] = []

    for output_index, record in enumerate(records):
        encoding = tokenizer.encode(record.text, add_special_tokens=False)
        if not encoding.ids:
            raise ValueError(f"Tokenizer produced no IDs for {record.assignment_id}")
        aligned_word_counts = token_aligned_word_counts(record.text, list(encoding.offsets))
        if sum(aligned_word_counts) != record.word_count:
            raise ValueError(
                f"Word alignment mismatch for {record.assignment_id}: "
                f"{sum(aligned_word_counts)} != {record.word_count}"
            )
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if decoded != record.text:
            raise ValueError(f"Tokenizer round trip failed for {record.assignment_id}")
        boundaries = [1 if count > 0 else 0 for count in aligned_word_counts]
        token_documents.append(torch.tensor(encoding.ids, dtype=token_dtype))
        word_count_documents.append(torch.tensor(aligned_word_counts, dtype=torch.int16))
        word_boundary_documents.append(torch.tensor(boundaries, dtype=torch.int8))
        token_lengths.append(len(encoding.ids))
        word_lengths.append(record.word_count)
        index_rows.append(
            {
                "output_index": output_index,
                "source_row": record.source_row,
                "assignment_id": record.assignment_id,
                "memory_type": record.memory_type,
                "words": record.word_count,
                "tokens": len(encoding.ids),
                "text_sha256": record.text_sha256,
            }
        )

    atomic_torch_save(token_documents, paths["tokenized_bin"])
    atomic_torch_save(word_count_documents, paths["word_counts"])
    atomic_torch_save(word_boundary_documents, paths["word_boundaries"])
    atomic_write_text(
        paths["index"],
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index_rows),
    )

    artifact_rows = {
        name: {
            "path": portable_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if name != "manifest"
    }
    manifest = {
        "format": "hippocorpus_babylm2026_strict_heldout_v1",
        "source": {
            "dataset": "Hippocorpus V3",
            "path": portable_path(source_csv),
            "sha256": sha256_file(source_csv),
            "citation": "Sap et al. (2020), ACL 2020, doi:10.18653/v1/2020.acl-main.178",
            "license_note": "Dataset license is separate from the Apache-2.0 Hugging Face loader; retain the official distribution terms.",
        },
        "selection": {
            "text_field": "story",
            "normalization": "HTML entity decoding, Unicode NFKC, control removal, whitespace collapse",
            "case_and_punctuation_preserved": True,
            "word_definition": "maximal whitespace-separated spans",
            "min_words": args.min_words,
            "max_words": args.max_words,
            "urls_allowed": args.allow_urls,
            "emails_allowed": args.allow_emails,
            "rejected": dict(sorted(rejected.items())),
        },
        "tokenizer": {
            "path": portable_path(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": vocab_size,
            "model_type": tokenizer_payload["model"]["type"],
            "normalizer": tokenizer_payload.get("normalizer"),
            "pre_tokenizer": tokenizer_payload.get("pre_tokenizer"),
        },
        "statistics": {
            "source_documents": len(records) + sum(rejected.values()),
            "kept_documents": len(records),
            "memory_types": dict(sorted(memory_types.items())),
            "total_words": sum(word_lengths),
            "total_tokens": sum(token_lengths),
            "tokens_per_word": sum(token_lengths) / sum(word_lengths),
            "words_per_document": {
                "min": min(word_lengths),
                "p50": percentile(word_lengths, 0.50),
                "p95": percentile(word_lengths, 0.95),
                "max": max(word_lengths),
            },
            "tokens_per_document": {
                "min": min(token_lengths),
                "p50": percentile(token_lengths, 0.50),
                "p95": percentile(token_lengths, 0.95),
                "max": max(token_lengths),
            },
            "round_trip_verified_documents": len(records),
        },
        "artifacts": artifact_rows,
        "analysis_defaults": {
            "language": "eng",
            "language_id": 0,
            "max_documents": 2000,
            "document_sampling": "reservoir",
            "windows_per_document": 1,
            "seed": 42,
        },
    }
    atomic_write_text(paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare(args)
    stats = manifest["statistics"]
    print(
        f"Prepared {stats['kept_documents']:,} Hippocorpus stories with "
        f"{stats['total_words']:,} words and {stats['total_tokens']:,} tokens."
    )
    print(f"Manifest: {args.output_prefix.with_name(f'{args.output_prefix.name}.meta.json')}")


if __name__ == "__main__":
    main()

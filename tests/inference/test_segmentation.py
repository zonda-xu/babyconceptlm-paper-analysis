from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from scripts.analyze_segmentation import (
    DocumentStats,
    JsonBackedTokenizer,
    aggregate_document_stats,
    bootstrap_metrics,
    classify_segmentation,
    dataset_specs_from_manifest,
    histogram_quantile,
    paired_documents,
)


class SegmentationClassificationTests(unittest.TestCase):
    def test_separates_learned_and_max_forced_boundaries(self) -> None:
        result = classify_segmentation(
            [0, 0, 1, 1, 1, 2],
            max_chunk_len=3,
            left_censored=False,
            right_censored=False,
        )

        self.assertEqual(result["lengths"], [2, 3, 1])
        self.assertEqual(result["boundary_positions"], [2, 5])
        self.assertEqual(result["learned_positions"], [2])
        self.assertEqual(result["max_forced_positions"], [5])

    def test_removes_window_censored_segments_from_primary_lengths(self) -> None:
        result = classify_segmentation(
            [0, 0, 1, 1, 2, 2],
            max_chunk_len=8,
            left_censored=True,
            right_censored=True,
        )

        self.assertEqual(result["lengths"], [2, 2, 2])
        self.assertEqual(result["complete_lengths"], [2])

    def test_histogram_quantile_uses_global_counts(self) -> None:
        histogram = Counter({1: 1, 2: 3, 8: 1})

        self.assertEqual(histogram_quantile(histogram, 0.5), 2.0)
        self.assertEqual(histogram_quantile(histogram, 0.9), 8.0)


class AggregateAndBootstrapTests(unittest.TestCase):
    def build_document(self, source_index: int, scale: int) -> DocumentStats:
        return DocumentStats(
            source_index=source_index,
            windows=1,
            tokens=12 * scale,
            concepts=3 * scale,
            complete_segments=3 * scale,
            length_histogram=Counter({4: 3 * scale}),
            complete_length_histogram=Counter({4: 3 * scale}),
            boundary_eligible=10 * scale,
            observed_boundaries=3 * scale,
            learned_boundaries=2 * scale,
            max_forced_boundaries=1 * scale,
            gate_sum=0.6 * scale,
            gate_square_sum=0.12 * scale,
            gate_count=3 * scale,
        )

    def test_micro_aggregates_use_counts_not_document_means(self) -> None:
        documents = [self.build_document(0, 1), self.build_document(1, 2)]

        metrics = aggregate_document_stats(documents, max_chunk_len=8)["metrics"]

        self.assertAlmostEqual(metrics["tokens_per_concept"], 4.0)
        self.assertAlmostEqual(metrics["learned_boundary_rate"], 0.2)
        self.assertAlmostEqual(metrics["max_forced_boundary_rate"], 0.1)
        self.assertAlmostEqual(metrics["mean_segment_length"], 4.0)

    def test_bootstrap_is_deterministic_and_document_level(self) -> None:
        documents = [self.build_document(index, index + 1) for index in range(4)]

        first = bootstrap_metrics(documents, 8, samples=50, confidence=0.95, seed=7)
        second = bootstrap_metrics(documents, 8, samples=50, confidence=0.95, seed=7)

        self.assertEqual(first, second)
        self.assertIn("tokens_per_concept", first)
        self.assertEqual(first["tokens_per_concept"]["samples"], 50)


class DataLoadingTests(unittest.TestCase):
    def test_json_tokenizer_fallback_decodes_bytelevel_tokens(self) -> None:
        tokenizer = JsonBackedTokenizer(
            source_path=Path("tokenizer.json"),
            tokenizer_json={
                "model": {
                    "vocab": {
                        "<pad>": 0,
                        "<s>": 1,
                        "</s>": 2,
                        "<unk>": 3,
                        "<mask>": 4,
                        "h": 5,
                        "i": 6,
                        "Ġ": 7,
                    }
                },
                "decoder": {"type": "ByteLevel"},
            },
        )

        self.assertEqual(tokenizer.decode([5, 6, 7, 5, 6]), "hi hi")

    def test_manifest_relocates_stale_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "eng_dev.bin"
            torch.save([torch.tensor([1, 2, 3])], data_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "languages": [
                            {
                                "name": "eng",
                                "id": 0,
                                "validation_path": f"/stale/server/path/{data_path.name}",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            specs = dataset_specs_from_manifest(manifest_path, "validation")

            self.assertEqual(specs[0].path, data_path)

    def test_sidecars_remain_aligned_with_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            token_path = root / "tokens.pt"
            boundary_path = root / "boundaries.pt"
            torch.save([torch.tensor([5, 6]), torch.tensor([7])], token_path)
            torch.save([torch.tensor([1, 0]), torch.tensor([1])], boundary_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "languages": [
                            {
                                "name": "zho",
                                "id": 2,
                                "validation_path": str(token_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            spec = dataset_specs_from_manifest(manifest_path, "validation")[0]
            spec = type(spec)(
                name=spec.name,
                path=spec.path,
                language_id=spec.language_id,
                word_boundary_path=boundary_path,
            )

            documents = list(paired_documents(spec))

            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].word_boundaries.tolist(), [1, 0])
            self.assertEqual(documents[1].tokens.tolist(), [7])


if __name__ == "__main__":
    unittest.main()

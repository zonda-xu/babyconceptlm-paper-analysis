from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders

from scripts.prepare_hippocorpus import clean_story, parse_args, prepare


class HippocorpusCleaningTests(unittest.TestCase):
    def test_normalizes_without_lowercasing_or_dropping_punctuation(self) -> None:
        self.assertEqual(clean_story("  A\u00a0Story!\nNext.  "), "A Story! Next.")


class HippocorpusPreparationTests(unittest.TestCase):
    def test_writes_aligned_token_and_word_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "stories.csv"
            with source_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["AssignmentId", "memType", "story"])
                writer.writeheader()
                writer.writerow(
                    {
                        "AssignmentId": "good",
                        "memType": "recalled",
                        "story": "This is a clean story. It preserves punctuation and case.",
                    }
                )
                writer.writerow(
                    {
                        "AssignmentId": "duplicate",
                        "memType": "imagined",
                        "story": "  This is a clean story. It preserves punctuation and case.  ",
                    }
                )
                writer.writerow(
                    {
                        "AssignmentId": "url",
                        "memType": "retold",
                        "story": "This otherwise valid story contains https://example.com in its text.",
                    }
                )
            output_prefix = root / "heldout"
            tokenizer_path = root / 'synthetic_tokenizer.json'
            alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
            tokenizer = Tokenizer(models.BPE(vocab={c:i for i,c in enumerate(alphabet)}, merges=[]))
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.save(str(tokenizer_path))
            args = parse_args(
                [
                    "--source_csv",
                    str(source_path),
                    "--tokenizer",
                    str(tokenizer_path),
                    "--output_prefix",
                    str(output_prefix),
                    "--min_words",
                    "1",
                ]
            )

            manifest = prepare(args)

            token_documents = torch.load(root / "heldout_tokenized.bin", weights_only=True)
            word_counts = torch.load(root / "heldout_word_counts.pt", weights_only=True)
            word_boundaries = torch.load(root / "heldout_word_boundaries.pt", weights_only=True)
            self.assertEqual(len(token_documents), 1)
            self.assertEqual(len(token_documents[0]), len(word_counts[0]))
            self.assertEqual(len(token_documents[0]), len(word_boundaries[0]))
            self.assertEqual(int(word_counts[0].sum()), 10)
            self.assertTrue(torch.equal((word_counts[0] > 0).to(torch.int8), word_boundaries[0]))
            self.assertEqual(manifest["selection"]["rejected"]["contains_url"], 1)
            self.assertEqual(manifest["selection"]["rejected"]["duplicate_normalized_story"], 1)
            loaded_manifest = json.loads((root / "heldout.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded_manifest["statistics"]["kept_documents"], 1)


if __name__ == "__main__":
    unittest.main()

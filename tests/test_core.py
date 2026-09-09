import csv
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from babyconceptlm_analysis.flops import concept_flops, reproduce_moments, analyze_counts
from babyconceptlm_analysis.reference import normalized_log_auc, validate_reference
from babyconceptlm_analysis.stats import paired_test, holm, analyze_csv
from babyconceptlm_analysis.cli import main

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def test_all_archived_aggregate_checks(self):
        result = validate_reference(ROOT/'examples/reference', ROOT/'configs/flops.json')
        self.assertEqual(result['status'], 'passed')
        self.assertGreater(result['check_count'], 200)
        self.assertAlmostEqual(result['flops']['chinese']['allocated_tokens_per_concept'], 2.94649565, places=6)
        self.assertAlmostEqual(result['flops']['chinese']['concept_ideal_at_mean_count'], 47.071703965, places=6)

    def test_log_auc(self):
        self.assertEqual(normalized_log_auc([1, 10, 100], [0, 50, 100]), 50)
        with self.assertRaises(ValueError):
            normalized_log_auc([1, 1], [0, 1])

    def test_quadratic_moment_identity(self):
        config = json.loads((ROOT/'configs/flops.json').read_text())['chinese']
        counts = np.array([3., 8., 40., 160.])
        expected = float(concept_flops(counts, config).mean())
        actual = float(concept_flops(counts.mean(), config)) + 8*config['num_concept_layers']*config['concept_size']*counts.var()
        self.assertAlmostEqual(actual, expected, places=4)

    def test_seeded_count_analysis_includes_partial_batch(self):
        config = json.loads((ROOT/'configs/flops.json').read_text())['chinese']
        result = analyze_counts([2, 8, 5, 2, 1], config, 2, 10, 3)
        self.assertEqual(result, analyze_counts([2, 8, 5, 2, 1], config, 2, 10, 3))
        self.assertEqual(result['sequences'], 5)
        with self.assertRaises(ValueError):
            analyze_counts([0, 4], config, 2, 10, 3)


class StatsTests(unittest.TestCase):
    def test_exact_sign_flip(self):
        result = paired_test([1, 1, 1], seed=0, samples=100)
        self.assertEqual(result['p_exact_two_sided'], .25)
        self.assertIsNone(result['dz'])
        self.assertEqual(result['bootstrap_percentile_95'], [1, 1])
        self.assertEqual(paired_test([0, 0], seed=0, samples=100)['p_exact_two_sided'], 1)

    def test_holm(self):
        np.testing.assert_allclose(holm([.04, .01, .03]), [.06, .03, .06])

    def test_synthetic_pair(self):
        result = analyze_csv(ROOT/'examples/fmri_synthetic.csv', 'synthetic_concept', 'synthetic_control', 17, samples=200)
        self.assertEqual(len(result['tasks']), 2)
        self.assertAlmostEqual(result['tasks']['synthetic_language']['delta'], 1)

    def test_duplicate_and_missing_subjects_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scores.csv'
            for rows in [
                [('a','t','1',.1),('a','t','1',.2)],
                [('a','t','1',.1),('b','t','2',.2)],
            ]:
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['model','task','participant','score'])
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    analyze_csv(path,'a','b',0,samples=100)


class CountsCliTests(unittest.TestCase):
    def test_completed_count_metadata_and_integrity_check(self):
        config = json.loads((ROOT/'configs/flops.json').read_text())['chinese']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'counts.csv'
            path.write_text('sequence_index,concept_count\n0,2\n1,8\n2,5\n')
            metadata = {'status':'complete','sequence_length':256,'sequences':3,
                        'counts_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'residual_gate':True,
                        'flops_shape':{k:v for k,v in config.items() if k not in {'include_residual_gate','token_control_layers','config_lm_concept_residual_gate'}}}
            meta = path.with_suffix('.meta.json')
            meta.write_text(json.dumps(metadata))
            output = Path(tmp)/'result.json'
            argv = ['analysis','flops-from-counts','--input',str(path),'--config',str(ROOT/'configs/flops.json'),
                    '--pair','chinese','--batch-size','2','--permutations','3','--seed','9','--output',str(output)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(json.loads(output.read_text())['input_completion_verified'])
            path.write_text('sequence_index,concept_count\n0,3\n1,8\n2,5\n')
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_requested_task_family_must_exist(self):
        with self.assertRaises(ValueError):
            analyze_csv(ROOT/'examples/fmri_synthetic.csv', 'synthetic_concept', 'synthetic_control', 0,
                        samples=100, tasks=['missing_task'])


if __name__ == '__main__':
    unittest.main()

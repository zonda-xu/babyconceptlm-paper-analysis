import json
import tempfile
import unittest
from pathlib import Path

import torch

from babyconceptlm_analysis._packed_data import PackedTokenizedDataset, MultiPackedTokenizedDataset, PackedSequenceCollator


class PackingTests(unittest.TestCase):
    def test_final_partial_sequence_and_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tokens.bin'
            torch.save([torch.tensor([3, 4, 5]), torch.tensor([6])], path)
            data = PackedTokenizedDataset(str(path), seq_length=4, eos_token_id=2, pad_token_id=0)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['input_ids'].tolist(), [3,4,5,2])
            # Frozen packing inserts EOS between documents, not after the last.
            self.assertEqual(data[1]['input_ids'].tolist(), [6,0,0,0])
            self.assertEqual(data[1]['attention_mask'].tolist(), [1,0,0,0])
            batch = PackedSequenceCollator()([data[0], data[1]])
            self.assertEqual(tuple(batch['input_ids'].shape), (2,4))

    def test_languages_pack_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [('eng',3),('zho',4)]:
                torch.save([torch.tensor([value])], root/f'{name}.bin')
            manifest = {'format':'multi1_multilingual_manifest_v1', 'languages':[
                {'name':'eng','id':0,'validation_path':'eng.bin'},
                {'name':'zho','id':2,'validation_path':'zho.bin'}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            data = MultiPackedTokenizedDataset(str(root/'manifest.json'), 'validation', 4, 2, 0)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['input_ids'].tolist(), [3,0,0,0])
            self.assertEqual(data[1]['input_ids'].tolist(), [4,0,0,0])


if __name__ == '__main__':
    unittest.main()

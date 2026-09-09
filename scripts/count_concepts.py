#!/usr/bin/env python3
"""Count deterministic concepts in every packed dev sequence, without training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data', type=Path, help='Tensor-list tokenized documents, loaded with weights_only=True.')
    inputs.add_argument('--manifest', type=Path, help='Multilingual validation manifest; paths relative to manifest.')
    parser.add_argument('--language-id', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept-remote-code', action='store_true')
    args = parser.parse_args()
    if not args.accept_remote_code:
        parser.error('Review local checkpoint code, then pass --accept-remote-code to allow its execution.')
    if args.batch_size < 1 or not args.checkpoint.is_dir():
        parser.error('A local checkpoint directory and positive batch size are required.')
    if args.output.exists() or args.output.with_suffix('.meta.json').exists():
        parser.error('Output exists; choose a new path to preserve earlier runs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_MODULES_CACHE'] = str((args.output.parent/'.hf_modules').resolve())
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM
    from babyconceptlm_analysis._boundary import count_segments, move_boundary_path
    from babyconceptlm_analysis._packed_data import MultiPackedTokenizedDataset, PackedTokenizedDataset, PackedSequenceCollator
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint.resolve()), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    model.eval()
    common = dict(seq_length=256, eos_token_id=int(model.config.eos_token_id),
                  pad_token_id=int(model.config.pad_token_id), pad_to_num_sequences_multiple=None)
    if args.manifest:
        dataset = MultiPackedTokenizedDataset(manifest_path=str(args.manifest.resolve()), split='validation', **common)
    else:
        dataset = PackedTokenizedDataset(path=str(args.data.resolve()), language_id=args.language_id, **common)
    if not len(dataset):
        raise ValueError('No packed sequences found.')
    move_boundary_path(model, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=PackedSequenceCollator())
    position = 0
    with args.output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence_index', 'concept_count'])
        for batch in loader:
            for count in count_segments(model, batch, device).tolist():
                writer.writerow([position, int(count)])
                position += 1
    data_path = args.manifest or args.data
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    metadata = {'status':'complete', 'sequence_length':256, 'sequences':position,
                'documents':int(dataset.num_documents), 'batch_size':args.batch_size,
                'device':str(device), 'dtype':str(model.dtype), 'torch_version':torch.__version__,
                'packing':'separate languages; EOS separators only when needed between documents; final partial sequence retained; no DDP padding',
                'config_sha256':digest(args.checkpoint/'config.json'),
                'input_sha256':digest(data_path), 'counts_sha256':digest(args.output),
                'input_hash_scope':'manifest only; separately retain hashes of every referenced shard and sidecar' if args.manifest else 'top-level token file; retain shard hashes if sharded',
                'warning':'A CSV from an interrupted run lacks this complete sidecar and must not be treated as full-dev.'}
    keys = ['hidden_size','intermediate_size','concept_size','concept_intermediate_size',
            'readout_intermediate_size','vocab_size','scan_size','boundary_syntax_dim',
            'num_token_layers','num_concept_layers','num_readout_layers','lm_concept_gate_hidden_size']
    metadata['flops_shape'] = {key:getattr(model.config,key) for key in keys}
    metadata['residual_gate'] = bool(getattr(model.config,'lm_concept_residual_gate',False))
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Counted all {position} packed sequences; no backbone/readout inference required.')


if __name__ == '__main__':
    main()

"""Read-only asset / runtime checks. Never train, run model inference or change historical assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
B_SHA = '50f306b0e2e50de49fe632377b8b4adcdd8c4f6eef50af3cff075a6a9b9415bd'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_path():
    found = shutil.which('git')
    if found:
        return found
    local = Path(os.environ['LOCALAPPDATA']) / 'GitHubDesktop'
    found = sorted(local.glob('app-*/resources/app/git/cmd/git.exe'))
    if found:
        return str(found[-1])
    standard = Path(os.environ['ProgramFiles']) / 'Git/cmd/git.exe'
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError('Git is required for training source snapshots')


def assets(data_root):
    evidence = data_root / 'audit/migration/harbor_transfer20261027_182041'
    derived = data_root / '_derived/harbor_poc_20261007_124509'
    git = git_path()
    def git_run(*args):
        return subprocess.check_output([git, '-C', str(ROOT), *args], text=True, encoding='utf-8').strip()
    def current_path(old):
        normalized = str(old).replace('\\', '/')
        data_prefix = 'D:/RSdataset/'
        if normalized.casefold().startswith(data_prefix.casefold()):
            return data_root / normalized[len(data_prefix):]
        old_repo = 'C:/Users/DOCTOR/Documents/GitHub/YOLOv8/'
        if normalized.casefold().startswith(old_repo.casefold()):
            return ROOT / normalized[len(old_repo):]
        return Path(old)

    errors, warnings, key_checks = [], [], []
    for item in load(evidence / 'key_sha256.json'):
        path = current_path(item['old_path'])
        exists = path.is_file()
        actual = sha(path) if exists else None
        valid = exists and actual == item['sha256'] and path.stat().st_size == item['bytes']
        key_checks.append({'path': str(path), 'required': item['requirement'], 'matches': valid, 'sha256': actual})
        if not valid:
            (errors if item['requirement'] == 'required' else warnings).append(f'Key asset missing or differs: {path}')

    inventory = []
    for entry in load(evidence / 'migration_manifest.json')['entries']:
        if entry['id'] not in ('DIOR', 'SAR', 'fixed_harbor'):
            continue
        path = current_path(entry['old_absolute_path'])
        files, size = 0, 0
        for folder, _, names in os.walk(path):
            for name in names:
                files += 1
                size += (Path(folder) / name).stat().st_size
        matches = (files, size) == (entry['files'], entry['bytes'])
        inventory.append({'id': entry['id'], 'files': files, 'bytes': size,
                          'expected_files': entry['files'], 'expected_bytes': entry['bytes'], 'matches': matches})
        if not matches:
            errors.append(f'Dataset inventory count/size differs: {path}')
    counts, references, seen_keys = {}, {}, set()
    hashes = load(derived / 'manifest_hashes.json')
    expected = {'train': {'rgb': (318, 1117), 'sar': (1290, 3199)},
                'val': {'rgb': (328, 1205), 'sar': (161, 391)},
                'test': {'rgb': (810, 3098), 'sar': (160, 385)}}
    for split in expected:
        path = derived / f'{split}_manifest.json'
        actual = sha(path)
        if hashes[path.name] != actual:
            errors.append(f'Manifest hash mismatch: {split}')
        records = load(path)
        counts[split] = {}
        for modality, target in expected[split].items():
            subset = [r for r in records if r['modality'] == modality]
            measured = (len(subset), sum(len(r['harbor']) for r in subset))
            counts[split][modality] = {'images': measured[0], 'boxes': measured[1]}
            if measured != target:
                errors.append(f'Count mismatch: {split}/{modality}')
        checked, missing = 0, []
        for record in records:
            if record['key'] in seen_keys:
                errors.append(f'Duplicate manifest key: {record["key"]}')
            seen_keys.add(record['key'])
            # Check the actual trainer paths, without applying a hidden remap.
            for field in ('image', 'annotation_source', 'label', 'original_image'):
                if record.get(field):
                    checked += 1
                    if not Path(record[field]).is_file():
                        missing.append({'key': record['key'], 'field': field, 'path': record[field]})
            for box in record['harbor']:
                x1, y1, x2, y2 = box['xyxy']
                if not (all(math.isfinite(v) for v in box['xyxy']) and
                        0 <= x1 < x2 <= record['width'] and 0 <= y1 < y2 <= record['height']):
                    errors.append(f'Invalid box: {record["key"]}')
        references[split] = {'checked': checked, 'missing': missing, 'sha256': actual}
        if missing:
            errors.append(f'{split}: {len(missing)} missing references')

    old_run = data_root / 'experiments/harbor_rgb_sar_20261024_212810_431389'
    source_checks = []
    for relative, expected_hash in load(evidence / 'resume_code_hashes.json').items():
        path, snapshot = ROOT / relative, old_run / 'source_snapshot' / relative
        exact = path.is_file() and sha(path) == expected_hash
        normalized = path.is_file() and snapshot.is_file() and (
            path.read_bytes().replace(b'\r\n', b'\n') == snapshot.read_bytes().replace(b'\r\n', b'\n'))
        source_checks.append({'path': relative, 'exact': exact, 'equal_after_newline_normalization': normalized})
    git_run('merge-base', '--is-ancestor', 'd1a300cd5064be6829d55f08d1e6f55bc69b7c19', 'HEAD')
    return {'passed': not errors, 'errors': errors, 'warnings': warnings,
            'git': {'executable': git, 'head': git_run('rev-parse', 'HEAD'),
                    'branch': git_run('branch', '--show-current'), 'status': git_run('status', '--short')},
            'key_assets': key_checks, 'dataset_inventory': inventory, 'counts': counts, 'references': references,
            'manifest_content_unchanged': all(r['matches'] for r in key_checks if '_manifest.json' in r['path']),
            'base_weight_mapping': {'old': r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt',
                                    'new': str(ROOT / 'ultralytics/best.pt')},
            'data_paths': 'Original D:/RSdataset namespace retained; no manifest edits',
            'historical_resume_source_checks': source_checks,
            'historical_resume_certified': False,
            'limits': ['Inventory checks counts/bytes, not every image content hash', 'No model inference, training or test evaluation',
                       'Old-run full resume identity has not been certified']}


def runtime(output, data_root):
    os.environ.update(YOLO_CONFIG_DIR=str(output / 'runtime_settings'), YOLO_OFFLINE='true',
                      YOLO_AUTOINSTALL='false', MPLCONFIGDIR=str(output / 'matplotlib'), WANDB_MODE='disabled')
    from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports
    setup_runtime()
    import torch
    import torchvision
    import cv2
    import numpy
    import matplotlib
    import PIL
    import yaml
    import psutil
    import ultralytics
    from tools.harbor_poc.core import load_base, detection_from_base, freeze, frozen_digest
    assert sha(ROOT / 'ultralytics/best.pt') == B_SHA
    base = load_base({'base_weights': str(ROOT / 'ultralytics/best.pt')})
    expert, inherited = detection_from_base(base)
    frozen = freeze(expert)
    assert inherited['new_random_parameters'] == 0
    assert frozen['trainable_parameters'] == 4085571 and frozen['total_parameters'] == 11135987
    assert_local_imports()
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Synthetic arithmetic/kernel checks only; no dataset/model inference.
    x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
    assert torch.equal((x @ torch.eye(4, device='cuda')).cpu(), x.cpu())
    conv = torch.nn.Conv2d(3, 4, 3).cuda()
    conv(torch.ones(1, 3, 16, 16, device='cuda')).sum().backward()
    assert torch.isfinite(conv.weight.grad).all()
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device='cuda'),
                               torch.tensor([.9, .8], device='cuda'), .5).cpu().tolist()
    assert kept == [0]
    torch.cuda.synchronize()
    # Exercise the same Windows spawn/shared-cache loader on a small fixed train subset.
    # No optimizer or model forward is involved.
    from tools.harbor_cli.data import HarborDataset, get_loader
    records = load(data_root / '_derived/harbor_poc_20261007_124509/train_manifest.json')
    sample = [r for mod in ('rgb', 'sar') for r in [r for r in records if r['modality'] == mod][:4]]
    dataset = HarborDataset(sample, 640, cache='shared')
    try:
        loader = get_loader(dataset, 4, 4, 2, True, True)
        batches = list(loader)
        assert sum(len(batch['indices']) for batch in batches) == 8
        assert all(batch['img'].shape == (4, 3, 640, 640) for batch in batches)
        assert all(torch.isfinite(batch['bboxes']).all() for batch in batches)
    finally:
        dataset.close()
    properties = torch.cuda.get_device_properties(0)
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    return {'passed': True, 'python': sys.version, 'executable': sys.executable, 'packages': packages,
            'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu': properties.name,
            'compute_capability': torch.cuda.get_device_capability(), 'compiled_arches': torch.cuda.get_arch_list(),
            'vram_bytes': properties.total_memory, 'ram_bytes': psutil.virtual_memory().total,
            'cpu_logical': psutil.cpu_count(), 'cpu_physical': psutil.cpu_count(logical=False),
            'disk_free_bytes': {drive: shutil.disk_usage(drive).free for drive in ('C:/', 'D:/')},
            'ultralytics_path': ultralytics.__file__, 'torch_path': torch.__file__,
            'base_model': type(base).__name__, 'base_classes': base.names,
            'expert_classes': expert.names, 'new_random_parameters': inherited['new_random_parameters'],
            'trainable_parameters': frozen['trainable_parameters'], 'frozen_digest': frozen_digest(expert),
            'data_loader': {'passed': True, 'samples': 8, 'workers': 4, 'cache': 'shared',
                            'persistent_workers': True, 'batch_size': 4},
            'cuda_matmul_conv_backward_nms': True, 'model_inference_performed': False,
            'training_started': False, 'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(r'D:\RSdataset'))
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_machine')
    parser.add_argument('--runtime', action='store_true', help='Check imports, CPU model conversion and synthetic CUDA kernels')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    name = 'runtime_check' if args.runtime else 'asset_check'
    try:
        result = runtime(args.output, args.data_root) if args.runtime else assets(args.data_root)
    except Exception:
        result = {'passed': False, 'error': traceback.format_exc()}
    path = args.output / f'{name}.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'report': str(path), 'errors': result.get('errors', result.get('error'))}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

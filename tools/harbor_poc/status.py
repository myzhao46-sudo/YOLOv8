"""Lightweight standard-library-only process and checkpoint status; no model loading."""
import argparse, ctypes, datetime, json, os
from ctypes import wintypes
from pathlib import Path

def read_json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError): return {}

def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, path)

def process_info(pid, expected_creation=None):
    result = {'pid': pid, 'alive': False}
    if not pid: return result
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        result['query_error'] = ctypes.get_last_error(); return result
    try:
        code = wintypes.DWORD()
        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            result.update(alive=code.value == 259, exit_code=code.value)
        times = [wintypes.FILETIME() for _ in range(4)]
        if kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            creation = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            result['creation_filetime'] = creation
            if expected_creation and creation != expected_creation:
                result.update(alive=False, pid_reused=True)
        in_job = wintypes.BOOL()
        if kernel.IsProcessInJob(handle, None, ctypes.byref(in_job)):
            result['in_windows_job'] = bool(in_job.value)
        else: result['job_query_error'] = ctypes.get_last_error()
    finally: kernel.CloseHandle(handle)
    return result

def compact_metrics(metrics):
    keys = ['images', 'gt', 'ap50', 'ap50_95', 'precision', 'recall', 'tp', 'fp', 'fn']
    return {m: {k: metrics[m][k] for k in keys if k in metrics[m]} for m in ['rgb', 'sar'] if m in metrics}

def snapshot(root):
    root = Path(root); active = read_json(root / 'active_process.json')
    run = active.get('run', 'A_seed42'); run_dir = root / 'runs' / run
    progress = read_json(run_dir / 'progress.json'); heartbeat = read_json(run_dir / 'heartbeat.json')
    result = read_json(run_dir / 'result.json'); best = read_json(run_dir / 'best_val_metrics.json')
    child = process_info(active.get('training_pid'), active.get('training_creation_filetime'))
    supervisor = process_info(active.get('supervisor_pid'), active.get('supervisor_creation_filetime'))
    error = None
    if child['alive']: status = 'running'
    elif active.get('status') == 'failed':
        status = 'failed'; code = active.get('exit_code')
        error = {'exit_code': code, 'exit_code_hex': f'0x{code & 0xffffffff:08X}' if isinstance(code, int) else None,
                 'detail': read_json(run_dir / 'failure.json').get('error', 'Native/process exit; inspect saved stderr and incident record.'),
                 'stderr': active.get('stderr')}
    elif result.get('status') == 'completed' and active.get('exit_code') == 0: status = 'completed'
    else:
        status = 'not_running_unconfirmed_exit'
        error = {'detail': 'Training process not alive; progress status alone is not authoritative.'}
    hb = heartbeat if heartbeat.get('pid') == active.get('training_pid') else {}
    last_metrics = progress.get('latest', {}).get('metrics', result.get('metrics', {}))
    return {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'status': status, 'run': run,
            'training_process': child, 'supervisor_process': supervisor,
            'last_completed_epoch': progress.get('last_completed_epoch', result.get('epochs')),
            'epoch_in_progress': hb.get('epoch_in_progress') if child['alive'] else None,
            'batch': hb.get('batch') if child['alive'] else None, 'heartbeat_time': hb.get('timestamp'),
            'best_epoch': progress.get('best_epoch', result.get('best_epoch')),
            'latest_validation': compact_metrics(last_metrics), 'best_validation': compact_metrics(best),
            'checkpoints': {k: str(run_dir / 'weights' / (k + '.pt')) for k in ['best', 'last']}, 'error': error}

def render(s):
    lines = [f"{s['status']} | {s['run']} | PID {s['training_process']['pid']} | {s['checked_at']}",
             f"epoch: completed={s['last_completed_epoch']} current={s['epoch_in_progress']} batch={s['batch']} best={s['best_epoch']}"]
    for label, key in [('latest', 'latest_validation'), ('best', 'best_validation')]:
        for mod, m in s[key].items():
            lines.append(f"{label} {mod.upper()}: AP50={m['ap50'] * 100:.3f}% AP50-95={m['ap50_95'] * 100:.3f}% "
                         f"P/R={m['precision'] * 100:.2f}/{m['recall'] * 100:.2f}% TP/FP/FN={m['tp']}/{m['fp']}/{m['fn']}")
    lines += [f"{k}: {v}" for k, v in s['checkpoints'].items()]
    lines.append('error: ' + (json.dumps(s['error'], ensure_ascii=False) if s['error'] else 'none reported'))
    return '\n'.join(lines)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--json', action='store_true'); a = p.parse_args()
    s = snapshot(a.root); print(json.dumps(s, ensure_ascii=False, indent=2) if a.json else render(s))

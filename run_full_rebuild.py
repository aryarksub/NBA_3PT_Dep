"""Rebuild the whole dataset from the SportVU archives, end to end.

Run from the repository root:

    .venv\\Scripts\\python.exe run_full_rebuild.py

Safe to interrupt and re-run. Moment conversion skips games already present in
data/moment_data/ and writes atomically, so an interrupted run loses at most the game in
flight. Later stages are gated by `redo` in src/pbp_shot_processing.py.

Refuses to start if another copy is already running, because two processes writing
data/moment_data/ at once wastes hours of duplicate work.
"""
import os
import subprocess
import sys
import time

PY = sys.executable
MOMENT_DIR = os.path.join('data', 'moment_data')
LOCK = os.path.join('data', '.rebuild.lock')

STAGES = [
    ('moment_processing   (7z -> moment CSVs)', os.path.join('src', 'moment_processing.py')),
    ('pbp_shot_processing (defender features)', os.path.join('src', 'pbp_shot_processing.py')),
    ('cf_dep_processing   (cross-fitted EP)',   os.path.join('src', 'cf_dep_processing.py')),
    ('cf_dependence       (metric + CIs)',      os.path.join('src', 'experiments', 'cf_dependence.py')),
    ('cf_ci_impact        (what CIs change)',   os.path.join('src', 'experiments', 'cf_ci_impact.py')),
    ('seed_sweep          (fold uncertainty)',  os.path.join('src', 'experiments', 'cf_dependence_seed_sweep.py')),
]


def another_copy_running():
    if not os.path.exists(LOCK):
        return False
    try:
        pid = int(open(LOCK).read().strip())
    except (ValueError, OSError):
        return False
    if pid == os.getpid():
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True   # cannot check; assume yes and let the user clear the lock


def main():
    if not os.path.isdir('src'):
        sys.exit('Run this from the repository root (the directory containing src/).')

    if another_copy_running():
        sys.exit(f'Another rebuild appears to be running (see {LOCK}). '
                 f'If that is wrong, delete {LOCK} and retry.')

    os.makedirs('data', exist_ok=True)
    with open(LOCK, 'w') as f:
        f.write(str(os.getpid()))

    done = len([f for f in os.listdir(MOMENT_DIR) if f.endswith('.csv')]) if os.path.isdir(MOMENT_DIR) else 0
    archives = len([f for f in os.listdir(os.path.join('data', 'game_logs')) if f.endswith('.7z')])
    print(f'{done} of {archives} games already converted; resuming.\n', flush=True)

    overall = time.time()
    try:
        for name, script in STAGES:
            print(f'\n{"="*70}\n=== {name}\n=== {time.strftime("%Y-%m-%d %H:%M:%S")}\n{"="*70}', flush=True)
            t = time.time()
            rc = subprocess.run([PY, script]).returncode
            if rc != 0:
                print(f'\n*** FAILED: {name} (exit {rc}) after {(time.time()-t)/60:.1f} min', flush=True)
                print('*** Completed stages are cached; fix and re-run.', flush=True)
                return rc
            print(f'\n--- {name}: {(time.time()-t)/60:.1f} min', flush=True)

        print(f'\n{"="*70}\nCOMPLETE in {(time.time()-overall)/3600:.2f} h\n{"="*70}', flush=True)
        print('Now set `redo = False` in src/pbp_shot_processing.py so later runs reuse the caches.',
              flush=True)
        return 0
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == '__main__':
    sys.exit(main())

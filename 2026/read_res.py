"""
read_res.py -- Grading script (for instructors / TAs)

Usage:
    python read_res.py <results directory>

Example:
    python read_res.py ./submissions/

The script scans the directory for all res2026_*.gzip files,
verifies the HMAC signature of each (tamper detection), and prints
a ranked leaderboard sorted by total score.
"""

import gzip, pickle, hmac, hashlib, os, sys
from functools import cmp_to_key
from datetime import datetime

_HMAC_KEY = b"ssl_pysim_2026_distributed_coop_challenge"


# -----------------------------------------------------------------------
# Load and verify a single result file
# -----------------------------------------------------------------------
def load_result(filepath):
    """
    Returns (data_dict, error_str).
    error_str is None if verification passed; otherwise describes the problem.
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            sig = f.read(32)   # first 32 bytes: HMAC-SHA256 signature
            raw = f.read()     # remainder: pickled data
    except Exception as e:
        return None, f"File read error: {e}"

    expected = hmac.new(_HMAC_KEY, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None, "HMAC mismatch (file may have been tampered with)"

    try:
        data = pickle.loads(raw)
    except Exception as e:
        return None, f"Data parse error: {e}"

    # Basic field check
    for field in ('student_id', 'seed', 'runs', 'total', 'ts'):
        if field not in data:
            return None, f"Missing field in result: {field}"

    # Consistency check
    if sum(data['runs']) != data['total']:
        return None, "total != sum(runs) (data may have been tampered with)"

    return data, None


# -----------------------------------------------------------------------
# Ranking comparison function
# -----------------------------------------------------------------------
def compare(item_a, item_b):
    """
    Ranking rules (higher is better):
      1. total score
      2. best single-run score (tie-break)
      3. worst single-run score (stability tie-break)
    """
    a, b = item_a[1], item_b[1]

    if a['total'] != b['total']:
        return b['total'] - a['total']

    ma, mb = max(a['runs']), max(b['runs'])
    if ma != mb:
        return mb - ma

    mina, minb = min(a['runs']), min(b['runs'])
    return minb - mina


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"[Error] Directory not found: {folder}")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith('.gzip') and f.startswith('res2026_')
    )

    if not files:
        print(f"[Info] No res2026_*.gzip files found in: {folder}")
        sys.exit(0)

    W = 65
    print(f"\n{'='*W}")
    print(f"  2026 Robotics Assessment -- Results Summary")
    print(f"{'='*W}")
    print(f"  Directory : {folder}")
    print(f"  Files found: {len(files)}")
    print(f"{'='*W}\n")

    results = []
    invalid = []

    for fname in files:
        path = os.path.join(folder, fname)
        data, err = load_result(path)
        if err:
            invalid.append((fname, err))
            print(f"  [INVALID] {fname:40s}  -> {err}")
            continue

        ts_str = datetime.fromtimestamp(data['ts']).strftime('%Y-%m-%d %H:%M:%S')
        results.append((fname, data))
        print(f"  [OK]      ID: {data['student_id']:<12}  "
              f"Total: {data['total']:>3}  "
              f"SEED: {data['seed']}  "
              f"Generated: {ts_str}")

    if not results:
        print("\nNo valid results. Exiting.")
        sys.exit(0)

    results.sort(key=cmp_to_key(compare))

    print(f"\n{'='*W}")
    print(f"  Final Rankings  ({len(results)} valid submissions)")
    print(f"{'='*W}")
    print(f"  {'Rank':>4}  {'Student ID':<13}  {'Total':>5}  "
          f"{'Best':>4}  {'Worst':>5}  Per-round scores")
    print(f"  {'-'*60}")

    for rank, (fname, d) in enumerate(results, 1):
        runs   = d['runs']
        marker = " *" if rank == 1 else "  "
        print(
            f"{marker}{rank:>3}.  "
            f"{d['student_id']:<13}  "
            f"{d['total']:>5}  "
            f"{max(runs):>4}  "
            f"{min(runs):>5}  "
            f"{runs}"
        )

    print(f"\n{'='*W}")
    print(f"  Ranking criteria:")
    print(f"    1. Total tasks completed across all 10 runs (higher is better)")
    print(f"    2. Best single-run score (tie-break)")
    print(f"    3. Worst single-run score (stability tie-break, higher is better)")
    print(f"{'='*W}\n")

    if invalid:
        print(f"[Note] {len(invalid)} invalid file(s) excluded from ranking:")
        for fname, err in invalid:
            print(f"  {fname}: {err}")
        print()

"""
read_res.py —— 评分脚本（助教 / 出题人使用）

用法：
    python read_res.py <结果文件目录>

示例：
    python read_res.py ./submissions/

脚本将扫描目录下所有 res2026_*.gzip 文件，
验证 HMAC 签名（防篡改），并输出按总分排名的成绩表。
"""

import gzip, pickle, hmac, hashlib, os, sys
from functools import cmp_to_key
from datetime import datetime

_HMAC_KEY = b"ssl_pysim_2026_distributed_coop_challenge"


# ──────────────────────────────────────────────────────────────────────
# 读取并验证单份结果文件
# ──────────────────────────────────────────────────────────────────────
def load_result(filepath):
    """
    返回 (data_dict, error_str)。
    error_str 为 None 表示验证通过；否则描述错误原因。
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            sig = f.read(32)   # 前32字节为 HMAC-SHA256 签名
            raw = f.read()     # 剩余为 pickle 数据
    except Exception as e:
        return None, f"文件读取失败: {e}"

    expected = hmac.new(_HMAC_KEY, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None, "HMAC 签名不匹配（文件可能被篡改）"

    try:
        data = pickle.loads(raw)
    except Exception as e:
        return None, f"数据解析失败: {e}"

    # 基本字段检查
    for field in ('student_id', 'seed', 'runs', 'total', 'ts'):
        if field not in data:
            return None, f"结果文件缺少字段: {field}"

    # 校验 total 与 runs 是否一致
    if sum(data['runs']) != data['total']:
        return None, "total 与 runs 之和不一致（数据可能被篡改）"

    return data, None


# ──────────────────────────────────────────────────────────────────────
# 排名比较函数
# ──────────────────────────────────────────────────────────────────────
def compare(item_a, item_b):
    """
    排名规则（越好排名越靠前）：
      1. 总分（total）：越高越好
      2. 最高单轮得分（max(runs)）：越高越好
      3. 最低单轮得分（min(runs)）：越高越好（稳定性）
    """
    a, b = item_a[1], item_b[1]

    if a['total'] != b['total']:
        return b['total'] - a['total']

    ma, mb = max(a['runs']), max(b['runs'])
    if ma != mb:
        return mb - ma

    mina, minb = min(a['runs']), min(b['runs'])
    return minb - mina


# ──────────────────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"[错误] 目录不存在: {folder}")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith('.gzip') and f.startswith('res2026_')
    )

    if not files:
        print(f"[提示] 目录中未找到 res2026_*.gzip 文件: {folder}")
        sys.exit(0)

    print(f"\n{'='*65}")
    print(f"  2026 机器人考核 —— 分布式协作任务分配  成绩汇总")
    print(f"{'='*65}")
    print(f"  结果目录: {folder}")
    print(f"  共找到 {len(files)} 份结果文件")
    print(f"{'='*65}\n")

    results = []
    invalid = []

    for fname in files:
        path = os.path.join(folder, fname)
        data, err = load_result(path)
        if err:
            invalid.append((fname, err))
            print(f"  [无效] {fname:40s}  → {err}")
            continue

        # 格式化时间戳
        ts_str = datetime.fromtimestamp(data['ts']).strftime('%Y-%m-%d %H:%M:%S')
        results.append((fname, data))
        print(f"  [有效] 学号: {data['student_id']:<12}  "
              f"总分: {data['total']:>3}  "
              f"SEED: {data['seed']}  "
              f"生成时间: {ts_str}")

    if not results:
        print("\n没有有效结果，退出。")
        sys.exit(0)

    # 按排名规则排序
    results.sort(key=cmp_to_key(compare))

    N_RUNS = len(results[0][1]['runs'])

    # 打印排名表
    print(f"\n{'='*65}")
    print(f"  最终排名（共 {len(results)} 人有效）")
    print(f"{'='*65}")
    header = f"{'名次':>4}  {'学号':<13}  {'总分':>4}  " \
             f"{'最高':>4}  {'最低':>4}  各轮成绩"
    print(f"  {header}")
    print(f"  {'-'*60}")

    for rank, (fname, d) in enumerate(results, 1):
        runs   = d['runs']
        marker = " ★" if rank == 1 else "  "
        print(
            f"{marker}{rank:>3}.  "
            f"{d['student_id']:<13}  "
            f"{d['total']:>4}  "
            f"{max(runs):>4}  "
            f"{min(runs):>4}  "
            f"{runs}"
        )

    print(f"\n{'='*65}")
    print(f"  排名说明：")
    print(f"    1. 10轮合计完成任务数（总分）越高越好")
    print(f"    2. 总分相同时，最高单轮得分越高越好")
    print(f"    3. 仍相同时，最低单轮得分越高越好（稳定性）")
    print(f"{'='*65}\n")

    if invalid:
        print(f"[注意] 以下 {len(invalid)} 份文件无效，未参与排名：")
        for fname, err in invalid:
            print(f"  {fname}: {err}")
        print()

"""
2026 机器人专业转专业考核 —— 分布式协作任务分配
仿真主程序（请勿修改此文件）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import random, time, os, sys, gzip, pickle, hmac, hashlib

# ======================================================================
# 仿真参数（请勿修改）
# ======================================================================
ROBOT_SPEED       = 2.0    # 机器人移动速度 (m/s)
TASK_DONE_RADIUS  = 0.35   # 任务完成判定半径 (m)
FIELD_W           = 10.0   # 场地宽度 (m)
FIELD_H           = 10.0   # 场地高度 (m)
SIM_TIME          = 30.0   # 单次仿真时长 (s)
DT                = 0.1    # 仿真时间步长 (s)
N_RUNS            = 10     # 重复测试次数

# 两个机器人的固定起始位置
ROBOT_STARTS = [(1.0, 1.0), (9.0, 1.0)]

# 任务点参数
INITIAL_TASKS    = 10   # 初始任务点数量
REFRESH_TRIGGER  = 3    # 剩余任务数不足此值时触发补充
REFRESH_BATCH    = 5    # 每次补充的任务点数量
MAX_TOTAL_TASKS  = 60   # 单次仿真最多生成的任务点总数（实际上限）

# 结果文件签名密钥（用于防止篡改）
_HMAC_KEY = b"ssl_pysim_2026_distributed_coop_challenge"

# 可视化颜色
_COLORS = ['#1565C0', '#C62828']   # 机器人A：蓝  机器人B：红
_LABELS = ['机器人 A', '机器人 B']


# ======================================================================
# Robot —— 机器人模型
# ======================================================================
class Robot:
    def __init__(self, start_x, start_y, robot_id):
        self.start = (start_x, start_y)
        self.id    = robot_id
        self.reset()

    def reset(self):
        self.x, self.y    = self.start
        self.theta        = np.pi / 2   # 初始朝向：正上方
        self.target_pos   = None        # 当前目标坐标 tuple(x, y)，None 表示无目标
        self.n_completed  = 0           # 本轮完成任务数
        self.traj         = [self.start]

    # ── 目标管理 ──────────────────────────────────────────────────────

    def get_target_idx(self, tasks):
        """在 tasks 中查找当前目标的下标；找不到（目标已消失）则返回 -1"""
        if self.target_pos is None:
            return -1
        for i, t in enumerate(tasks):
            if abs(t[0] - self.target_pos[0]) < 1e-9 and \
               abs(t[1] - self.target_pos[1]) < 1e-9:
                return i
        return -1

    def set_target(self, idx, tasks):
        """设置目标；idx=-1 或越界时清空目标"""
        if 0 <= idx < len(tasks):
            self.target_pos = tasks[idx]
        else:
            self.target_pos = None

    def clear_target(self):
        self.target_pos = None

    # ── 运动 ──────────────────────────────────────────────────────────

    def move(self, dt):
        """向当前目标移动一步"""
        if self.target_pos is None:
            return
        dx = self.target_pos[0] - self.x
        dy = self.target_pos[1] - self.y
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return
        self.theta = np.arctan2(dy, dx)
        step = min(ROBOT_SPEED * dt, dist)
        self.x += step * np.cos(self.theta)
        self.y += step * np.sin(self.theta)
        self.traj.append((self.x, self.y))

    def dist_to_target(self):
        if self.target_pos is None:
            return float('inf')
        return np.hypot(self.x - self.target_pos[0],
                        self.y - self.target_pos[1])

    @property
    def pos(self):
        return (self.x, self.y)


# ======================================================================
# TaskManager —— 任务点管理
# ======================================================================
class TaskManager:
    def __init__(self):
        self._tasks         = []   # list of (x, y)
        self.total_spawned  = 0
        self.total_completed = 0

    def reset(self):
        self._tasks          = []
        self.total_spawned   = 0
        self.total_completed = 0
        self._spawn(INITIAL_TASKS)

    def _spawn(self, n):
        added = 0
        while added < n and self.total_spawned < MAX_TOTAL_TASKS:
            x = random.uniform(0.5, FIELD_W - 0.5)
            y = random.uniform(0.5, FIELD_H - 0.5)
            self._tasks.append((x, y))
            self.total_spawned += 1
            added += 1

    def get_tasks(self):
        """返回当前未完成任务列表的拷贝"""
        return list(self._tasks)

    def maybe_refresh(self):
        """任务不足时补充"""
        if len(self._tasks) <= REFRESH_TRIGGER \
                and self.total_spawned < MAX_TOTAL_TASKS:
            self._spawn(REFRESH_BATCH)

    def try_complete(self, robot_pos):
        """
        尝试完成 robot_pos 处机器人能触达的任务（先调用先得）。
        返回本次完成数。
        """
        x, y = robot_pos
        completed = 0
        remaining = []
        for t in self._tasks:
            if np.hypot(x - t[0], y - t[1]) < TASK_DONE_RADIUS:
                completed += 1
                self.total_completed += 1
            else:
                remaining.append(t)
        self._tasks = remaining
        return completed


# ======================================================================
# Visualizer —— 可视化
# ======================================================================
class Visualizer:
    def __init__(self, sim_pause_time):
        self.spt    = sim_pause_time
        self.paused = False
        self._build_fig()

    def _build_fig(self):
        self.fig = plt.figure(figsize=(17, 7))
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[1.5, 1],
                               hspace=0.40, wspace=0.30)
        # 左侧：主仿真视图（跨上下两行）
        self.ax_sim  = self.fig.add_subplot(gs[:, 0])
        # 右上：完成进度折线
        self.ax_prog = self.fig.add_subplot(gs[0, 1])
        # 右下：文字信息面板
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_info.axis('off')

        plt.ion()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._ts  = []   # 时间序列
        self._cs  = []   # 累计完成序列
        self._flash_tasks = []  # 刚完成的任务闪烁（暂未使用）

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused

    def reset_run(self):
        self._ts = []
        self._cs = []

    def update(self, robots, task_mgr, t, run_idx):
        tasks = task_mgr.get_tasks()
        total_done = sum(r.n_completed for r in robots)

        # ── 左侧：主仿真 ──────────────────────────────────────────────
        ax = self.ax_sim
        ax.clear()
        ax.set_xlim(-0.5, FIELD_W + 0.5)
        ax.set_ylim(-0.5, FIELD_H + 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#F8F8F0')

        # 场地边框
        rect = patches.FancyBboxPatch(
            (0, 0), FIELD_W, FIELD_H,
            boxstyle='square,pad=0',
            linewidth=2, edgecolor='#444', facecolor='#FFFDE7')
        ax.add_patch(rect)
        ax.set_title(
            f'Run {run_idx + 1} / {N_RUNS}    t = {t:.1f} s',
            fontsize=13, fontweight='bold')

        # 任务点（灰色圆点）
        if tasks:
            tx, ty = zip(*tasks)
            ax.scatter(tx, ty, c='#9E9E9E', s=55, zorder=3,
                       marker='o', label='待完成任务')

        # 机器人
        for robot in robots:
            c = _COLORS[robot.id]
            # 历史轨迹
            traj = robot.traj[-150:]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=c, alpha=0.22, linewidth=1.8)
            # 车身圆圈
            body = patches.Circle(
                (robot.x, robot.y), 0.38,
                color=c, zorder=5, alpha=0.88)
            ax.add_patch(body)
            ax.text(robot.x, robot.y, str(robot.id),
                    ha='center', va='center',
                    color='white', fontweight='bold',
                    fontsize=11, zorder=6)
            # 目标连线与目标圆
            tidx = robot.get_target_idx(tasks)
            if tidx >= 0:
                gx, gy = tasks[tidx]
                ax.annotate(
                    '', xy=(gx, gy), xytext=(robot.x, robot.y),
                    arrowprops=dict(
                        arrowstyle='->', color=c,
                        lw=1.6, alpha=0.55))
                ax.add_patch(patches.Circle(
                    (gx, gy), TASK_DONE_RADIUS,
                    color=c, fill=False,
                    lw=1.5, linestyle='--', alpha=0.6, zorder=4))
            # 标签
            ax.text(robot.x + 0.45, robot.y + 0.45,
                    f'{_LABELS[robot.id]}\n完成: {robot.n_completed}',
                    fontsize=8, color=c, zorder=7)

        # 图例
        legend_elems = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLORS[0], markersize=10,
                   label=_LABELS[0]),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLORS[1], markersize=10,
                   label=_LABELS[1]),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#9E9E9E', markersize=8,
                   label='待完成任务'),
        ]
        ax.legend(handles=legend_elems, loc='upper left', fontsize=8)
        ax.set_xlabel(
            f'已完成: {total_done}  |  剩余: {len(tasks)}'
            f'  |  总生成: {task_mgr.total_spawned}',
            fontsize=9)

        # ── 右上：进度折线 ────────────────────────────────────────────
        self._ts.append(t)
        self._cs.append(total_done)
        ax2 = self.ax_prog
        ax2.clear()
        ax2.plot(self._ts, self._cs, color='#1565C0', linewidth=2)
        ax2.set_xlim(0, SIM_TIME)
        ax2.set_xlabel('时间 (s)', fontsize=9)
        ax2.set_ylabel('累计完成任务数', fontsize=9)
        ax2.set_title('完成进度', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ── 右下：文字信息面板 ────────────────────────────────────────
        ax3 = self.ax_info
        ax3.clear()
        ax3.axis('off')
        info = (
            f"Run {run_idx + 1} / {N_RUNS}\n"
            f"时间: {t:.1f} / {SIM_TIME:.0f} s\n"
            f"\n"
            f"机器人A 完成: {robots[0].n_completed}\n"
            f"机器人B 完成: {robots[1].n_completed}\n"
            f"合计完成: {total_done}\n"
            f"\n"
            f"剩余任务: {len(tasks)}\n"
            f"总生成:   {task_mgr.total_spawned}\n"
            f"\n"
            f"[空格] 暂停 / 继续"
        )
        ax3.text(
            0.08, 0.96, info,
            transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6',
                      facecolor='#E8EAF6', alpha=0.85))

        # ── 暂停控制 ──────────────────────────────────────────────────
        if self.spt < 0:
            self.paused = True
            while self.paused:
                plt.pause(0.05)
                if not plt.fignum_exists(self.fig.number):
                    break
        else:
            plt.pause(self.spt)

    def close(self):
        plt.close(self.fig)


# ======================================================================
# 单次仿真
# ======================================================================
def run_once(robots, task_mgr, choose_task_fn, visualizer, run_idx):
    """
    执行一次完整仿真（SIM_TIME 秒），返回两机器人合计完成的任务数。
    """
    for robot in robots:
        robot.reset()
    task_mgr.reset()
    if visualizer:
        visualizer.reset_run()

    total_completed = 0

    for step in range(int(SIM_TIME / DT)):
        t = step * DT
        tasks = task_mgr.get_tasks()

        # 如果任务为空先补充
        if not tasks:
            task_mgr.maybe_refresh()
            tasks = task_mgr.get_tasks()

        # ── 步骤1：使失效目标无效化 ───────────────────────────────────
        for robot in robots:
            if robot.target_pos is not None \
                    and robot.get_target_idx(tasks) == -1:
                robot.clear_target()   # 目标已被队友完成，清空

        # ── 步骤2：获取"本步"各机器人目标下标（用于传给对方）────────
        target_idxs = [r.get_target_idx(tasks) for r in robots]

        # ── 步骤3：需要新目标的机器人同步调用策略函数 ────────────────
        #   两机器人互相使用对方"上一时刻"的目标（分布式、无中心调度）
        new_choices = [None, None]
        for i, robot in enumerate(robots):
            if target_idxs[i] == -1 and tasks:
                partner = robots[1 - i]
                try:
                    new_idx = choose_task_fn(
                        robot.pos,
                        robot.id,
                        partner.pos,
                        target_idxs[1 - i],
                        list(tasks)          # 传拷贝，防止策略函数修改
                    )
                except Exception as e:
                    print(f"\n[警告] 机器人{i} choose_task 抛出异常: {e}")
                    new_idx = 0
                # 合法性检查
                if new_idx is None or not (0 <= new_idx < len(tasks)):
                    new_idx = 0
                new_choices[i] = int(new_idx)

        # ── 步骤4：设置新目标 ────────────────────────────────────────
        for i, robot in enumerate(robots):
            if new_choices[i] is not None:
                robot.set_target(new_choices[i], tasks)

        # ── 步骤5：机器人移动 ────────────────────────────────────────
        for robot in robots:
            robot.move(DT)

        # ── 步骤6：任务完成检测（先处理 A，再处理 B → 先到先得）───────
        for robot in robots:
            done = task_mgr.try_complete(robot.pos)
            if done:
                robot.n_completed += done
                total_completed   += done
                robot.clear_target()   # 完成后重新选择

        # ── 步骤7：任务补充 ──────────────────────────────────────────
        task_mgr.maybe_refresh()

        # ── 步骤8：可视化 ────────────────────────────────────────────
        if visualizer:
            visualizer.update(robots, task_mgr, t, run_idx)

    return total_completed


# ======================================================================
# 结果加密保存
# ======================================================================
def save_result(student_id, seed, run_scores):
    """
    将测试结果序列化、签名后保存为 res2026_<学号>.gzip。
    签名使用 HMAC-SHA256，防止文件内容被篡改。
    """
    data = {
        'student_id': student_id,
        'seed':       seed,
        'runs':       run_scores,          # list[int]，长度=N_RUNS
        'total':      sum(run_scores),
        'ts':         int(time.time()),    # 生成时间戳
    }
    raw = pickle.dumps(data, protocol=4)
    sig = hmac.new(_HMAC_KEY, raw, hashlib.sha256).digest()   # 32字节签名

    fname = f"res2026_{student_id}.gzip"
    with gzip.open(fname, 'wb') as f:
        f.write(sig)    # 前32字节为签名
        f.write(raw)    # 后续为数据

    return fname


# ======================================================================
# 主函数
# ======================================================================
if __name__ == '__main__':
    # 支持 PyInstaller 打包后的路径
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_path)

    # 导入学生策略文件
    try:
        from strategy import choose_task, STUDENT_ID, SEED, SIM_PAUSE_TIME
    except ImportError as e:
        print(f"[错误] 无法导入 strategy.py: {e}")
        sys.exit(1)

    # 校验学号
    if not STUDENT_ID or STUDENT_ID.strip() in ('', '请填写你的学号'):
        print("[错误] 请先在 strategy.py 中将 STUDENT_ID 改为你的真实学号")
        sys.exit(1)

    # 初始化随机数
    _seed = SEED if SEED is not None else random.randrange(10000)
    random.seed(_seed)
    np.random.seed(_seed)

    print("=" * 50)
    print(f"  2026 机器人专业考核 —— 分布式协作任务分配")
    print("=" * 50)
    print(f"  学号  : {STUDENT_ID}")
    print(f"  SEED  : {_seed}")
    print(f"  运行  : {N_RUNS} 轮，每轮 {SIM_TIME:.0f} 秒")
    print("=" * 50)

    robots   = [Robot(*ROBOT_STARTS[0], 0), Robot(*ROBOT_STARTS[1], 1)]
    task_mgr = TaskManager()
    vis      = Visualizer(SIM_PAUSE_TIME)

    run_scores = []
    for run_idx in range(N_RUNS):
        print(f"\nRun {run_idx + 1:>2}/{N_RUNS} ... ", end='', flush=True)
        score = run_once(robots, task_mgr, choose_task, vis, run_idx)
        run_scores.append(score)
        print(f"完成 {score:>2} 个任务")

    vis.close()

    total = sum(run_scores)
    print("\n" + "=" * 50)
    print(f"  学号    : {STUDENT_ID}")
    print(f"  SEED    : {_seed}")
    print(f"  总分    : {total} 个任务（{N_RUNS} 轮合计）")
    print(f"  各轮得分: {run_scores}")
    print("=" * 50)

    fname = save_result(STUDENT_ID, _seed, run_scores)
    print(f"\n结果文件已保存: {fname}")
    print("请将此文件提交给助教，命名格式：学号_姓名_res2026.gzip")

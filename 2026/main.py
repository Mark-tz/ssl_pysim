"""
2026 机器人专业转专业考核 —— 分布式协作任务分配
仿真主程序（请勿修改此文件）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import importlib.util
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
MAX_TOTAL_TASKS  = 60   # 单次仿真最多生成的任务点总数

# 结果文件签名密钥（用于防止篡改）
_HMAC_KEY = b"ssl_pysim_2026_distributed_coop_challenge"

# 可视化颜色
_COLORS = ['#1565C0', '#C62828']   # 机器人A：蓝  机器人B：红
_LABELS = ['机器人 A', '机器人 B']

_W = 56   # 终端输出宽度


# ======================================================================
# 终端工具函数
# ======================================================================
def _line(char='═'):
    return char * _W

def _box_line(text='', char=' '):
    """在宽度 _W 内居中对齐，两侧加 ║"""
    inner = _W - 2
    return '║' + text.center(inner) + '║'

def print_banner(student_id, seed, strategy_path):
    print()
    print('╔' + '═' * (_W - 2) + '╗')
    print(_box_line('2026 机器人专业考核'))
    print(_box_line('分布式协作任务分配'))
    print('╠' + '═' * (_W - 2) + '╣')
    print(_box_line(f'学号 : {student_id}'))
    print(_box_line(f'SEED : {seed}'))
    print(_box_line(f'轮数 : {N_RUNS} 轮 × {SIM_TIME:.0f} 秒/轮'))
    print('╚' + '═' * (_W - 2) + '╝')
    print()
    print(f'  [√] strategy.py 路径 : {strategy_path}')
    print(f'  [√] STUDENT_ID       : {student_id}')
    print(f'  [√] SEED             : {seed}')
    print()

def print_run_result(run_idx, score, robot_a_done, robot_b_done, elapsed):
    bar_total = 20
    # 用完成数相对最大值画进度条（仅视觉效果，以MAX_TOTAL_TASKS为满）
    filled = min(bar_total, int(bar_total * score / MAX_TOTAL_TASKS))
    bar = '█' * filled + '░' * (bar_total - filled)
    print(
        f'  Run {run_idx+1:>2}/{N_RUNS}  [{bar}]  '
        f'完成: {score:>3}  '
        f'(A:{robot_a_done:>2} B:{robot_b_done:>2})  '
        f'{elapsed:>5.1f}s'
    )

def print_summary(student_id, seed, run_scores, run_details, fname):
    scores = np.array(run_scores)
    best_idx  = int(np.argmax(scores))
    worst_idx = int(np.argmin(scores))
    print()
    print(_line('═'))
    print('  测试完成！各轮明细：')
    print(_line('─'))
    print(f'  {"Run":>4}  {"总完成":>6}  {"A":>4}  {"B":>4}  {"耗时":>6}')
    print(f'  {"─"*4}  {"─"*6}  {"─"*4}  {"─"*4}  {"─"*6}')
    for i, (sc, da, db, et) in enumerate(run_details):
        marker = ' ★' if i == best_idx else ('  ' if i != worst_idx else ' ↓')
        print(f'{marker} {i+1:>3}.  {sc:>6}  {da:>4}  {db:>4}  {et:>5.1f}s')
    print(_line('─'))
    print(f'  各轮得分: {run_scores}')
    print(f'  最高单轮: {scores.max():.0f}  （第 {best_idx+1} 轮）')
    print(f'  最低单轮: {scores.min():.0f}  （第 {worst_idx+1} 轮）')
    print(f'  平均得分: {scores.mean():.1f}')
    print(f'  标准差  : {scores.std():.1f}')
    print(f'  总   分: {scores.sum():.0f}  （{N_RUNS} 轮合计）')
    print(_line('─'))
    print(f'  学号    : {student_id}')
    print(f'  SEED    : {seed}')
    print(_line('═'))
    print()
    print(f'  结果文件已保存: {fname}')
    print(f'  提交时请重命名为: 学号_姓名_res2026.gzip')
    print()


# ======================================================================
# strategy.py 加载（使用 importlib 按绝对路径显式加载，绕过 PyInstaller
# 可能引入的模块缓存与 sys.path 问题）
# ======================================================================
def _pause_exit(code=1):
    """退出前等待用户按键（仅在 exe 双击环境有效；脚本/管道环境直接退出）"""
    try:
        input('按回车键退出...')
    except (EOFError, OSError):
        pass
    sys.exit(code)


def load_strategy(app_path):
    """
    从 app_path 目录显式加载 strategy.py。
    无论 PyInstaller 是否已将某个版本的 strategy 打入包内，
    此函数始终读取磁盘上的外部文件，保证学生修改即时生效。
    """
    strategy_path = os.path.join(app_path, 'strategy.py')

    # ── 文件存在性检查 ──────────────────────────────────────────────
    if not os.path.isfile(strategy_path):
        print()
        print('[错误] 找不到 strategy.py，请检查以下内容：')
        print(f'       期望路径: {strategy_path}')
        print(f'       请确认 strategy.py 与 main.exe 在同一目录下。')
        print()
        _pause_exit()

    # ── 显式加载 ────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location('strategy_external', strategy_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as e:
        print()
        print('[错误] strategy.py 存在语法错误，无法执行：')
        print(f'       {e}')
        print()
        _pause_exit()
    except Exception as e:
        print()
        print(f'[错误] strategy.py 执行时抛出异常：{e}')
        import traceback; traceback.print_exc()
        print()
        _pause_exit()

    # ── 必要属性检查 ────────────────────────────────────────────────
    missing = [attr for attr in ('STUDENT_ID', 'SEED', 'SIM_PAUSE_TIME', 'choose_task')
               if not hasattr(mod, attr)]
    if missing:
        print()
        print(f'[错误] strategy.py 缺少以下必要定义: {missing}')
        print('       请使用题目提供的原始 strategy.py 模板。')
        print()
        _pause_exit()

    # ── choose_task 可调用性检查 ────────────────────────────────────
    if not callable(mod.choose_task):
        print()
        print('[错误] strategy.py 中的 choose_task 不是一个函数。')
        print()
        _pause_exit()

    # ── 学号检查 ────────────────────────────────────────────────────
    sid = str(getattr(mod, 'STUDENT_ID', '')).strip()
    if not sid or sid == '请填写你的学号':
        print()
        print('[错误] 请先在 strategy.py 中填写你的学号！')
        print(f'       当前值: "{sid}"')
        print(f'       修改方法: 将 strategy.py 第 7 行改为  STUDENT_ID = "你的学号"')
        print(f'       strategy.py 路径: {strategy_path}')
        print()
        _pause_exit()

    return mod, strategy_path


# ======================================================================
# Robot —— 机器人模型
# ======================================================================
class Robot:
    def __init__(self, start_x, start_y, robot_id):
        self.start = (start_x, start_y)
        self.id    = robot_id
        self.reset()

    def reset(self):
        self.x, self.y   = self.start
        self.theta        = np.pi / 2
        self.target_pos   = None
        self.n_completed  = 0
        self.traj         = [self.start]

    def get_target_idx(self, tasks):
        if self.target_pos is None:
            return -1
        for i, t in enumerate(tasks):
            if abs(t[0] - self.target_pos[0]) < 1e-9 and \
               abs(t[1] - self.target_pos[1]) < 1e-9:
                return i
        return -1

    def set_target(self, idx, tasks):
        if 0 <= idx < len(tasks):
            self.target_pos = tasks[idx]
        else:
            self.target_pos = None

    def clear_target(self):
        self.target_pos = None

    def move(self, dt):
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

    @property
    def pos(self):
        return (self.x, self.y)


# ======================================================================
# TaskManager —— 任务点管理
# ======================================================================
class TaskManager:
    def __init__(self):
        self._tasks          = []
        self.total_spawned   = 0
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
        return list(self._tasks)

    def maybe_refresh(self):
        if len(self._tasks) <= REFRESH_TRIGGER \
                and self.total_spawned < MAX_TOTAL_TASKS:
            self._spawn(REFRESH_BATCH)

    def try_complete(self, robot_pos):
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
    def __init__(self, sim_pause_time, student_id, seed):
        self.spt        = sim_pause_time
        self.paused     = False
        self.student_id = student_id
        self.seed       = seed
        self._build_fig()

    def _build_fig(self):
        self.fig = plt.figure(figsize=(17, 7))
        self.fig.patch.set_facecolor('#F0F0F0')
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[1.5, 1],
                               hspace=0.40, wspace=0.30)
        self.ax_sim  = self.fig.add_subplot(gs[:, 0])
        self.ax_prog = self.fig.add_subplot(gs[0, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_info.axis('off')

        plt.ion()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._ts = []
        self._cs = []

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused

    def reset_run(self):
        self._ts = []
        self._cs = []

    def update(self, robots, task_mgr, t, run_idx, run_scores_so_far):
        tasks      = task_mgr.get_tasks()
        total_done = sum(r.n_completed for r in robots)

        # ── 主仿真视图 ────────────────────────────────────────────────
        ax = self.ax_sim
        ax.clear()
        ax.set_xlim(-0.5, FIELD_W + 0.5)
        ax.set_ylim(-0.5, FIELD_H + 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#FFFDE7')

        rect = patches.FancyBboxPatch(
            (0, 0), FIELD_W, FIELD_H,
            boxstyle='square,pad=0',
            linewidth=2, edgecolor='#555', facecolor='#FFFDE7')
        ax.add_patch(rect)

        title_str = (f'Run {run_idx+1}/{N_RUNS}    '
                     f't = {t:.1f} s    '
                     f'SEED = {self.seed}')
        ax.set_title(title_str, fontsize=12, fontweight='bold')

        if tasks:
            tx, ty = zip(*tasks)
            ax.scatter(tx, ty, c='#9E9E9E', s=55, zorder=3, marker='o')

        for robot in robots:
            c = _COLORS[robot.id]
            traj = robot.traj[-150:]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=c, alpha=0.22, linewidth=1.8)
            ax.add_patch(patches.Circle(
                (robot.x, robot.y), 0.38, color=c, zorder=5, alpha=0.88))
            ax.text(robot.x, robot.y, str(robot.id),
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=11, zorder=6)
            tidx = robot.get_target_idx(tasks)
            if tidx >= 0:
                gx, gy = tasks[tidx]
                ax.annotate('', xy=(gx, gy), xytext=(robot.x, robot.y),
                            arrowprops=dict(arrowstyle='->',
                                            color=c, lw=1.6, alpha=0.55))
                ax.add_patch(patches.Circle(
                    (gx, gy), TASK_DONE_RADIUS,
                    color=c, fill=False, lw=1.5,
                    linestyle='--', alpha=0.6, zorder=4))
            ax.text(robot.x + 0.45, robot.y + 0.45,
                    f'{_LABELS[robot.id]}\n完成: {robot.n_completed}',
                    fontsize=8, color=c, zorder=7)

        legend_elems = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=_COLORS[0], markersize=10, label=_LABELS[0]),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=_COLORS[1], markersize=10, label=_LABELS[1]),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#9E9E9E', markersize=8,  label='待完成任务'),
        ]
        ax.legend(handles=legend_elems, loc='upper left', fontsize=8)
        ax.set_xlabel(
            f'已完成: {total_done}  |  剩余: {len(tasks)}'
            f'  |  总生成: {task_mgr.total_spawned}',
            fontsize=9)

        # ── 进度折线 ──────────────────────────────────────────────────
        self._ts.append(t)
        self._cs.append(total_done)
        ax2 = self.ax_prog
        ax2.clear()
        ax2.plot(self._ts, self._cs, color='#1565C0', linewidth=2)
        ax2.set_xlim(0, SIM_TIME)
        ax2.set_xlabel('时间 (s)', fontsize=9)
        ax2.set_ylabel('累计完成任务数', fontsize=9)
        ax2.set_title(f'Run {run_idx+1} 完成进度', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ── 文字信息面板 ──────────────────────────────────────────────
        ax3 = self.ax_info
        ax3.clear()
        ax3.axis('off')

        # 已完成轮次摘要
        prev_str = ''
        if run_scores_so_far:
            prev_str = '  '.join(str(s) for s in run_scores_so_far)

        info = (
            f"学号: {self.student_id}\n"
            f"SEED: {self.seed}\n"
            f"\n"
            f"Run {run_idx+1} / {N_RUNS}\n"
            f"时间: {t:.1f} / {SIM_TIME:.0f} s\n"
            f"\n"
            f"机器人A 完成: {robots[0].n_completed}\n"
            f"机器人B 完成: {robots[1].n_completed}\n"
            f"合计完成:   {total_done}\n"
            f"\n"
            f"剩余任务: {len(tasks)}\n"
            f"总生成:   {task_mgr.total_spawned}\n"
        )
        if run_scores_so_far:
            info += f"\n已完成轮得分:\n[{prev_str}]"
        info += "\n\n[空格] 暂停 / 继续"

        ax3.text(0.08, 0.97, info,
                 transform=ax3.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#E8EAF6', alpha=0.88))

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
def run_once(robots, task_mgr, choose_task_fn, visualizer,
             run_idx, run_scores_so_far):
    for robot in robots:
        robot.reset()
    task_mgr.reset()
    if visualizer:
        visualizer.reset_run()

    total_completed = 0
    t0 = time.time()

    for step in range(int(SIM_TIME / DT)):
        t = step * DT
        tasks = task_mgr.get_tasks()

        if not tasks:
            task_mgr.maybe_refresh()
            tasks = task_mgr.get_tasks()

        # 步骤1：失效目标清空
        for robot in robots:
            if robot.target_pos is not None \
                    and robot.get_target_idx(tasks) == -1:
                robot.clear_target()

        # 步骤2：获取当前目标下标
        target_idxs = [r.get_target_idx(tasks) for r in robots]

        # 步骤3：同步调用策略（使用对方上一时刻目标，模拟分布式）
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
                        list(tasks)
                    )
                except Exception as e:
                    print(f'\n  [警告] Run {run_idx+1} 步骤{step}: '
                          f'机器人{i} choose_task 异常: {e}')
                    new_idx = 0
                if new_idx is None or not (0 <= new_idx < len(tasks)):
                    new_idx = 0
                new_choices[i] = int(new_idx)

        # 步骤4：设置新目标
        for i, robot in enumerate(robots):
            if new_choices[i] is not None:
                robot.set_target(new_choices[i], tasks)

        # 步骤5：移动
        for robot in robots:
            robot.move(DT)

        # 步骤6：任务完成（A优先，先到先得）
        for robot in robots:
            done = task_mgr.try_complete(robot.pos)
            if done:
                robot.n_completed += done
                total_completed   += done
                robot.clear_target()

        # 步骤7：任务补充
        task_mgr.maybe_refresh()

        # 步骤8：可视化
        if visualizer:
            visualizer.update(robots, task_mgr, t, run_idx, run_scores_so_far)

    elapsed = time.time() - t0
    return total_completed, robots[0].n_completed, robots[1].n_completed, elapsed


# ======================================================================
# 结果加密保存
# ======================================================================
def save_result(student_id, seed, run_scores):
    data = {
        'student_id': student_id,
        'seed':       seed,
        'runs':       run_scores,
        'total':      sum(run_scores),
        'ts':         int(time.time()),
    }
    raw = pickle.dumps(data, protocol=4)
    sig = hmac.new(_HMAC_KEY, raw, hashlib.sha256).digest()

    fname = f'res2026_{student_id}.gzip'
    with gzip.open(fname, 'wb') as f:
        f.write(sig)
        f.write(raw)
    return fname


# ======================================================================
# 主函数
# ======================================================================
if __name__ == '__main__':

    # ── 确定 app 目录 ─────────────────────────────────────────────────
    # sys.frozen=True 时说明是 PyInstaller 打包后的 exe，
    # sys.executable 指向 exe 文件本身，dirname 即 exe 所在目录。
    # 否则为直接执行 main.py，使用 __file__ 所在目录。
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))

    # ── 加载 strategy.py（显式按路径加载，不依赖 sys.path）────────────
    mod, strategy_path = load_strategy(app_path)

    STUDENT_ID    = mod.STUDENT_ID.strip()
    SEED          = mod.SEED
    SIM_PAUSE_TIME = mod.SIM_PAUSE_TIME
    choose_task   = mod.choose_task

    # ── 随机种子初始化 ────────────────────────────────────────────────
    _seed = SEED if SEED is not None else random.randrange(10000)
    random.seed(_seed)
    np.random.seed(_seed)

    # ── 启动横幅 ─────────────────────────────────────────────────────
    print_banner(STUDENT_ID, _seed, strategy_path)
    print(f'  {_line("─")}')
    print(f'  开始测试，共 {N_RUNS} 轮，每轮 {SIM_TIME:.0f} 秒...')
    print(f'  {_line("─")}')
    print()

    # ── 初始化仿真对象 ────────────────────────────────────────────────
    robots   = [Robot(*ROBOT_STARTS[0], 0), Robot(*ROBOT_STARTS[1], 1)]
    task_mgr = TaskManager()
    vis      = Visualizer(SIM_PAUSE_TIME, STUDENT_ID, _seed)

    # ── 主循环 ────────────────────────────────────────────────────────
    run_scores  = []
    run_details = []   # (total, a_done, b_done, elapsed)

    for run_idx in range(N_RUNS):
        score, a_done, b_done, elapsed = run_once(
            robots, task_mgr, choose_task, vis,
            run_idx, run_scores
        )
        run_scores.append(score)
        run_details.append((score, a_done, b_done, elapsed))
        print_run_result(run_idx, score, a_done, b_done, elapsed)

    vis.close()

    # ── 保存结果 ──────────────────────────────────────────────────────
    fname = save_result(STUDENT_ID, _seed, run_scores)

    # ── 汇总输出 ──────────────────────────────────────────────────────
    print_summary(STUDENT_ID, _seed, run_scores, run_details, fname)

    # 防止 Windows 下 exe 双击后窗口立即关闭
    if getattr(sys, 'frozen', False):
        input('按回车键退出...')

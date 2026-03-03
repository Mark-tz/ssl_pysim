"""
2026 Robotics Major Assessment -- Distributed Cooperative Task Allocation
Main simulation program (do not modify this file)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import importlib.util
import random, time, os, sys, gzip, pickle, hmac, hashlib

# ======================================================================
# Simulation parameters (do not modify)
# ======================================================================
ROBOT_SPEED       = 2.0    # robot speed (m/s)
TASK_DONE_RADIUS  = 0.35   # task completion radius (m)
FIELD_W           = 10.0   # field width (m)
FIELD_H           = 10.0   # field height (m)
SIM_TIME          = 30.0   # simulation duration per run (s)
DT                = 0.1    # simulation time step (s)
N_RUNS            = 10     # number of test runs

# Fixed starting positions for the two robots
ROBOT_STARTS = [(1.0, 1.0), (9.0, 1.0)]

# Task parameters
INITIAL_TASKS    = 10   # tasks spawned at the start of each run
REFRESH_TRIGGER  = 3    # spawn new tasks when remaining count drops below this
REFRESH_BATCH    = 5    # number of tasks added per refresh
MAX_TOTAL_TASKS  = 60   # maximum tasks spawned in one run

# HMAC key for result file integrity (tamper detection)
_HMAC_KEY = b"ssl_pysim_2026_distributed_coop_challenge"

# 可视化颜色
_COLORS = ['#1565C0', '#C62828']   # 机器人A：蓝  机器人B：红
_LABELS = ['Robot A', 'Robot B']

_W = 56   # terminal output width


# ======================================================================
# Terminal helper functions
# ======================================================================
def _line(char='='):
    return char * _W

def _box_line(text=''):
    inner = _W - 2
    return '|' + text.center(inner) + '|'

def print_banner(student_id, seed, strategy_path):
    print()
    print('+' + '-' * (_W - 2) + '+')
    print(_box_line('2026 Robotics Assessment'))
    print(_box_line('Distributed Cooperative Task Allocation'))
    print('+' + '-' * (_W - 2) + '+')
    print(_box_line(f'ID   : {student_id}'))
    print(_box_line(f'SEED : {seed}'))
    print(_box_line(f'Runs : {N_RUNS} x {SIM_TIME:.0f}s each'))
    print('+' + '-' * (_W - 2) + '+')
    print()
    print(f'  [v] strategy.py : {strategy_path}')
    print(f'  [v] Student ID  : {student_id}')
    print(f'  [v] SEED        : {seed}')
    print()

def print_run_result(run_idx, score, robot_a_done, robot_b_done, elapsed):
    bar_total = 20
    filled = min(bar_total, int(bar_total * score / MAX_TOTAL_TASKS))
    bar = '#' * filled + '.' * (bar_total - filled)
    print(
        f'  Run {run_idx+1:>2}/{N_RUNS}  [{bar}]  '
        f'Done: {score:>3}  '
        f'(A:{robot_a_done:>2} B:{robot_b_done:>2})  '
        f'{elapsed:>5.1f}s'
    )

def print_summary(student_id, seed, run_scores, run_details, fname):
    scores = np.array(run_scores)
    best_idx  = int(np.argmax(scores))
    worst_idx = int(np.argmin(scores))
    print()
    print(_line('='))
    print('  Test complete!  Per-round details:')
    print(_line('-'))
    print(f'  {"Run":>4}  {"Done":>6}  {"A":>4}  {"B":>4}  {"Time":>6}')
    print(f'  {"----":>4}  {"------":>6}  {"----":>4}  {"----":>4}  {"------":>6}')
    for i, (sc, da, db, et) in enumerate(run_details):
        marker = ' *' if i == best_idx else ('  ' if i != worst_idx else ' v')
        print(f'{marker} {i+1:>3}.  {sc:>6}  {da:>4}  {db:>4}  {et:>5.1f}s')
    print(_line('-'))
    print(f'  Per-round  : {run_scores}')
    print(f'  Best run   : {int(scores.max())}  (Run {best_idx+1})')
    print(f'  Worst run  : {int(scores.min())}  (Run {worst_idx+1})')
    print(f'  Average    : {scores.mean():.1f}')
    print(f'  Std dev    : {scores.std():.1f}')
    print(f'  Total      : {int(scores.sum())}  ({N_RUNS} runs)')
    print(_line('-'))
    print(f'  Student ID : {student_id}')
    print(f'  SEED       : {seed}')
    print(_line('='))
    print()
    print(f'  Result saved : {fname}')
    print(f'  Rename to    : StudentID_Name_res2026.gzip')
    print()


# ======================================================================
# Load strategy.py via importlib (explicit absolute path).
# This bypasses PyInstaller module caching entirely -- the on-disk file
# is always read, so student edits take effect immediately.
# ======================================================================
def _pause_exit(code=1):
    """Wait for keypress before exiting (works in double-click exe; silent in pipe)."""
    try:
        input('Press Enter to exit...')
    except (EOFError, OSError):
        pass
    sys.exit(code)


def load_strategy(app_path):
    strategy_path = os.path.join(app_path, 'strategy.py')

    # -- file existence check ------------------------------------------
    if not os.path.isfile(strategy_path):
        print()
        print('[Error] Cannot find strategy.py. Please check:')
        print(f'        Expected path : {strategy_path}')
        print(f'        Make sure strategy.py is in the same folder as main.exe.')
        print()
        _pause_exit()

    # -- explicit load -------------------------------------------------
    spec = importlib.util.spec_from_file_location('strategy_external', strategy_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as e:
        print()
        print('[Error] Syntax error in strategy.py:')
        print(f'        {e}')
        print()
        _pause_exit()
    except Exception as e:
        print()
        print(f'[Error] Exception while loading strategy.py: {e}')
        import traceback; traceback.print_exc()
        print()
        _pause_exit()

    # -- required attribute check --------------------------------------
    missing = [a for a in ('STUDENT_ID', 'SEED', 'SIM_PAUSE_TIME', 'choose_task')
               if not hasattr(mod, a)]
    if missing:
        print()
        print(f'[Error] strategy.py is missing required definitions: {missing}')
        print('        Please use the original strategy.py template.')
        print()
        _pause_exit()

    if not callable(mod.choose_task):
        print()
        print('[Error] choose_task in strategy.py is not callable.')
        print()
        _pause_exit()

    # -- student ID check ----------------------------------------------
    sid = str(getattr(mod, 'STUDENT_ID', '')).strip()
    if not sid or sid == 'Fill in your student ID':
        print()
        print('[Error] Student ID not set in strategy.py!')
        print(f'        Current value : "{sid}"')
        print(f'        Fix : set  STUDENT_ID = "YourID"  on line 7 of strategy.py')
        print(f'        strategy.py path : {strategy_path}')
        print()
        _pause_exit()

    return mod, strategy_path


# ======================================================================
# Robot model
# ======================================================================
class Robot:
    def __init__(self, start_x, start_y, robot_id):
        self.start = (start_x, start_y)
        self.id    = robot_id
        self.reset()

    def reset(self):
        self.x, self.y   = self.start
        self.theta        = np.pi / 2   # initial heading: up
        self.target_pos   = None        # current target (x,y), None = no target
        self.n_completed  = 0           # tasks completed this run
        self.traj         = [self.start]

    def get_target_idx(self, tasks):
        """Return index of current target in tasks list; -1 if not found."""
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
# Task manager
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
        """Complete tasks within reach of robot_pos. First caller wins."""
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
# Visualiser
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

        # -- main simulation view ------------------------------------
        ax = self.ax_sim
        ax.clear()
        ax.set_xlim(-0.5, FIELD_W + 0.5)
        ax.set_ylim(-0.5, FIELD_H + 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#FFFDE7')

        ax.add_patch(patches.FancyBboxPatch(
            (0, 0), FIELD_W, FIELD_H,
            boxstyle='square,pad=0',
            linewidth=2, edgecolor='#555', facecolor='#FFFDE7'))

        ax.set_title(
            f'Run {run_idx+1}/{N_RUNS}    t = {t:.1f} s    SEED = {self.seed}',
            fontsize=12, fontweight='bold')

        if tasks:
            tx, ty = zip(*tasks)
            ax.scatter(tx, ty, c='#9E9E9E', s=55, zorder=3,
                       marker='o', label='Pending tasks')

        for robot in robots:
            c = _COLORS[robot.id]
            traj = robot.traj[-150:]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=c, alpha=0.22, linewidth=1.8)
            ax.add_patch(patches.Circle(
                (robot.x, robot.y), 0.38,
                color=c, zorder=5, alpha=0.88))
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
                    f'{_LABELS[robot.id]}\nDone: {robot.n_completed}',
                    fontsize=8, color=c, zorder=7)

        legend_elems = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=_COLORS[0], markersize=10, label=_LABELS[0]),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=_COLORS[1], markersize=10, label=_LABELS[1]),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#9E9E9E', markersize=8,  label='Pending Tasks'),
        ]
        ax.legend(handles=legend_elems, loc='upper left', fontsize=8)
        ax.set_xlabel(
            f'Done: {total_done}  |  Remaining: {len(tasks)}'
            f'  |  Spawned: {task_mgr.total_spawned}',
            fontsize=9)

        # -- progress line chart -------------------------------------
        self._ts.append(t)
        self._cs.append(total_done)
        ax2 = self.ax_prog
        ax2.clear()
        ax2.plot(self._ts, self._cs, color='#1565C0', linewidth=2)
        ax2.set_xlim(0, SIM_TIME)
        ax2.set_xlabel('Time (s)', fontsize=9)
        ax2.set_ylabel('Cumulative Completed Tasks', fontsize=9)
        ax2.set_title(f'Run {run_idx+1} Progress', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # -- text info panel -----------------------------------------
        ax3 = self.ax_info
        ax3.clear()
        ax3.axis('off')

        prev_str = '  '.join(str(s) for s in run_scores_so_far) \
                   if run_scores_so_far else ''

        info = (
            f"Student ID: {self.student_id}\n"
            f"SEED: {self.seed}\n"
            f"\n"
            f"Run {run_idx+1} / {N_RUNS}\n"
            f"Time: {t:.1f} / {SIM_TIME:.0f} s\n"
            f"\n"
            f"Robot A done: {robots[0].n_completed}\n"
            f"Robot B done: {robots[1].n_completed}\n"
            f"Total done:   {total_done}\n"
            f"\n"
            f"Remaining tasks: {len(tasks)}\n"
            f"Total spawned:   {task_mgr.total_spawned}\n"
        )
        if run_scores_so_far:
            info += f"\nCompleted run scores:\n[{prev_str}]"
        info += "\n\n[Space] Pause / Resume"

        ax3.text(0.08, 0.97, info,
                 transform=ax3.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#E8EAF6', alpha=0.88))

        # -- pause control -------------------------------------------
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
# Single run
# ======================================================================
def run_once(robots, task_mgr, choose_task_fn, visualizer,
             run_idx, run_scores_so_far, vis_stride=1):
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

        # Step 1: invalidate stale targets
        for robot in robots:
            if robot.target_pos is not None \
                    and robot.get_target_idx(tasks) == -1:
                robot.clear_target()

        # Step 2: capture current target indices
        target_idxs = [r.get_target_idx(tasks) for r in robots]

        # Step 3: robots that need a new target call choose_task simultaneously
        #         each uses the partner's *previous* target (distributed, no central scheduler)
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
                    print(f'\n  [Warning] Run {run_idx+1} step {step}: '
                          f'Robot {i} choose_task error: {e}')
                    new_idx = 0
                if new_idx is None or not (0 <= new_idx < len(tasks)):
                    new_idx = 0
                new_choices[i] = int(new_idx)

        # Step 4: apply new targets
        for i, robot in enumerate(robots):
            if new_choices[i] is not None:
                robot.set_target(new_choices[i], tasks)

        # Step 5: move
        for robot in robots:
            robot.move(DT)

        # Step 6: task completion (Robot A first -- first-come-first-served)
        for robot in robots:
            done = task_mgr.try_complete(robot.pos)
            if done:
                robot.n_completed += done
                total_completed   += done
                robot.clear_target()

        # Step 7: refresh tasks if needed
        task_mgr.maybe_refresh()

        # 步骤8：可视化
        if visualizer and (step % vis_stride == 0 or step == int(SIM_TIME / DT) - 1):
            visualizer.update(robots, task_mgr, t, run_idx, run_scores_so_far)

    elapsed = time.time() - t0
    return total_completed, robots[0].n_completed, robots[1].n_completed, elapsed


# ======================================================================
# Save encrypted result file
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
# Entry point
# ======================================================================
if __name__ == '__main__':

    # Resolve app directory:
    #   sys.frozen=True  -> PyInstaller exe; sys.executable points to the exe file
    #   sys.frozen=False -> running as script; use __file__
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))

    # Load strategy.py from disk (explicit path, bypasses PyInstaller cache)
    mod, strategy_path = load_strategy(app_path)

    STUDENT_ID     = mod.STUDENT_ID.strip()
    SEED           = mod.SEED
    SIM_PAUSE_TIME = mod.SIM_PAUSE_TIME
    choose_task   = mod.choose_task
    exam_mode     = bool(getattr(mod, 'EXAM_MODE', False))

    # Seed random number generators
    _seed = SEED if SEED is not None else random.randrange(10000)
    random.seed(_seed)
    np.random.seed(_seed)

    # Print startup banner
    print_banner(STUDENT_ID, _seed, strategy_path)
    print(f'  {_line("-")}')
    print(f'  Starting {N_RUNS} rounds, {SIM_TIME:.0f}s each...')
    print(f'  {_line("-")}')
    print()

    # Initialise simulation objects
    robots   = [Robot(*ROBOT_STARTS[0], 0), Robot(*ROBOT_STARTS[1], 1)]
    task_mgr = TaskManager()
    vis      = None if exam_mode else Visualizer(SIM_PAUSE_TIME, STUDENT_ID, _seed)

    if exam_mode:
        vis_stride = 1
    elif SIM_PAUSE_TIME < 0:
        vis_stride = 1
    elif SIM_PAUSE_TIME <= 0.01:
        vis_stride = 5
    elif SIM_PAUSE_TIME <= 0.02:
        vis_stride = 3
    else:
        vis_stride = 1

    # Main loop
    run_scores  = []
    run_details = []

    for run_idx in range(N_RUNS):
        score, a_done, b_done, elapsed = run_once(
            robots, task_mgr, choose_task, vis,
            run_idx, run_scores, vis_stride
        )
        run_scores.append(score)
        run_details.append((score, a_done, b_done, elapsed))
        print_run_result(run_idx, score, a_done, b_done, elapsed)

    if vis:
        vis.close()

    # Save result and print summary
    fname = save_result(STUDENT_ID, _seed, run_scores)
    print_summary(STUDENT_ID, _seed, run_scores, run_details, fname)

    # Keep console window open when launched by double-click on Windows
    if getattr(sys, 'frozen', False):
        _pause_exit(0)

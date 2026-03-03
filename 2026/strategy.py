# ======================================================================
# 2026 Robotics Assessment -- Distributed Cooperative Task Allocation
# Student answer file
# ======================================================================

# ★★★ 请将下方学号改为你的真实学号，否则无法生成有效结果文件 ★★★
STUDENT_ID = "Fill in your student ID"   # e.g. "3250101234"

# SEED: enter the number announced at the exam venue for the official test.
# During practice you can set any integer (e.g. 42, 100) or keep None (random map each run).
SEED = None

# Animation speed:
#   0.01  -> fast (recommended for debugging)
#   0.05  -> medium, easier to observe details
#   -1    -> manual step mode (press Space to advance one frame)
SIM_PAUSE_TIME = 0.01

# 考试模式（极速模式）：
#   True  → 关闭可视化界面，仅输出最终结果（速度最快，推荐正式跑分时使用）
#   False → 显示可视化界面（调试时使用）
EXAM_MODE = False

# ======================================================================
import numpy as np


def choose_task(my_pos, my_id, partner_pos, partner_target, tasks):
    """
    Task selection function -- implement your cooperative strategy here.

    This function is called once whenever a robot finishes a task or has
    no current target at startup.  Return the index of the task you want
    to head towards.

    Parameters
    ----------
    my_pos         : tuple (x, y)  -- my current position (metres)
    my_id          : int (0 or 1)  -- my robot ID  (A=0, B=1)
    partner_pos    : tuple (x, y)  -- partner's current position
    partner_target : int           -- index of the task the partner is
                                      heading to (-1 = partner has no target)
    tasks          : list[(x, y)]  -- positions of all incomplete tasks

    Returns
    -------
    int -- index into tasks of the task you want to complete
           (valid range: 0 .. len(tasks)-1)

    Notes
    -----
    * If both robots choose the same task, only the first to arrive scores;
      the other re-selects on the next call.
    * The tasks list changes dynamically: completed tasks are removed and
      new ones are added over time.
    * Both robots use the same function; they are distinguished by my_id.
    * partner_target reflects the partner's decision from the *previous*
      time step (slight information delay -- fully distributed).
    * Useful numpy helpers: np.hypot, np.argmin, np.array, np.where, etc.
    """
    if not tasks:
        return -1

    # ----------------------------------------------------------------
    # Write your strategy here (replace the default implementation below)
    # ----------------------------------------------------------------

    # Default strategy: go to the nearest task (greedy, ignores partner)
    my_pos = np.array(my_pos)
    dists = [np.hypot(my_pos[0] - t[0], my_pos[1] - t[1]) for t in tasks]
    return int(np.argmin(dists))

    # ----------------------------------------------------------------

import numpy as np

SEED = None
SIM_PAUSE_TIME = 0.01

def cost_fn(vel_x, vel_w, obstacle_min_dist, goal_dist):
    """
    输入参数：
        vel_x: 速度
        vel_w: 角速度
        obstacle_min_dist: 障碍物最小距离
        goal_dist: 到目标的距离
    """
    return goal_dist
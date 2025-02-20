import numpy as np
class DWAPlanner:
    def __init__(self, robot, dt, predict_time=1.0, v_resolution=7, w_resolution=5):
        self.robot = robot
        self.dt = dt
        self.predict_time = predict_time
        self.v_resolution = v_resolution
        self.w_resolution = w_resolution
        
        # 运动学约束
        self.max_speed = robot.max_speed
        self.max_angular_speed = robot.max_angular_speed
        self.max_acc = robot.max_acc
        self.max_angular_acc = robot.max_angular_acc
        
    def predict_trajectory(self, v, w):
        """预测给定速度下的轨迹"""
        trajectory = []
        state = np.array([self.robot.x, self.robot.y, self.robot.theta])
        
        for _ in np.arange(0, self.predict_time, self.dt):
            state = self.motion_model(state, v, w)
            trajectory.append(state)
            
        return np.array(trajectory)
    
    def motion_model(self, state, v, w):
        """运动学模型"""
        x, y, theta = state
        theta += w * self.dt
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        return np.array([x, y, theta])
    
    def plan(self, goal, obstacles, COST_FN):
        """DWA主算法"""
        # 根据当前速度和加速度限制生成速度网格
        v_min = max(-self.max_speed, self.robot.vx - self.max_acc * self.dt)
        v_max = min(self.max_speed, self.robot.vx + self.max_acc * self.dt)
        v_samples = np.linspace(v_min, v_max, self.v_resolution)
        
        w_min = max(-self.max_angular_speed, self.robot.vw - self.max_angular_acc * self.dt)
        w_max = min(self.max_angular_speed, self.robot.vw + self.max_angular_acc * self.dt)
        w_samples = np.linspace(w_min, w_max, self.w_resolution)
        
        # 创建速度组合矩阵
        v_grid, w_grid = np.meshgrid(v_samples, w_samples)
        v_grid = v_grid.ravel()
        w_grid = w_grid.ravel()
        
        # 向量化预测轨迹
        trajectories = self.predict_trajectories(v_grid, w_grid)
        
        # 向量化计算代价
        obstacle_min_dist = self.calculate_obstacle_dists(trajectories, obstacles)
        goal_dist = self.calculate_goal_dists(trajectories, goal)
        total_costs = COST_FN(v_grid, w_grid, obstacle_min_dist, goal_dist)
        
        # 找到最优速度
        min_idx = np.argmin(total_costs)
        best_v = v_grid[min_idx]
        best_w = w_grid[min_idx]
        
        return best_v, best_w, v_grid, w_grid, trajectories, total_costs
    
    def predict_trajectories(self, v_grid, w_grid):
        """向量化预测轨迹"""
        num_samples = len(v_grid)
        time_steps = int(self.predict_time / self.dt)
        
        # 初始化状态矩阵
        states = np.zeros((num_samples, time_steps, 3))
        states[:, 0, :] = [self.robot.x, self.robot.y, self.robot.theta]
        
        # 向量化更新状态
        for t in range(1, time_steps):
            states[:, t, 0] = states[:, t-1, 0] + v_grid * np.cos(states[:, t-1, 2]) * self.dt
            states[:, t, 1] = states[:, t-1, 1] + v_grid * np.sin(states[:, t-1, 2]) * self.dt
            states[:, t, 2] = states[:, t-1, 2] + w_grid * self.dt
            
        return states
    
    def calculate_obstacle_dists(self, trajectories, obstacles):
        """向量化计算障碍物代价"""
        obstacles = np.array(obstacles)
        num_samples, time_steps, _ = trajectories.shape
        
        # 计算所有点到所有障碍物的距离
        distances = np.zeros((num_samples, time_steps, len(obstacles)))
        for i in range(len(obstacles)):
            distances[:, :, i] = np.hypot(
                trajectories[:, :, 0] - obstacles[i, 0],
                trajectories[:, :, 1] - obstacles[i, 1]
            )
        
        # 找到每个轨迹的最小距离
        min_distances = np.min(distances, axis=(1, 2))
        return min_distances
        # return 1.0 / (min_distances + 1e-6)
    
    def calculate_goal_dists(self, trajectories, goal):
        """向量化计算目标代价"""
        final_positions = trajectories[:, -1, :2]
        return np.hypot(final_positions[:, 0] - goal[0],
                       final_positions[:, 1] - goal[1])
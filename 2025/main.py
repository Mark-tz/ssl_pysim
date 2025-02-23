import numpy as np
import matplotlib.pyplot as plt
import random

class RobotModel:
    def __init__(self, max_speed=4.0, max_angular_speed=2.0, max_acc=4, max_angular_acc=4):
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.max_acc = max_acc
        self.max_angular_acc = max_angular_acc
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vw = 0.0

    def update(self, dt, target_vx, target_vw):
        # Apply acceleration constraints
        dvx = np.clip(target_vx - self.vx, -self.max_acc*dt, self.max_acc*dt)
        dvw = np.clip(target_vw - self.vw, -self.max_angular_acc*dt, self.max_angular_acc*dt)
        
        # Update velocities with constraints
        self.vx = np.clip(self.vx + dvx, -self.max_speed, self.max_speed)
        self.vw = np.clip(self.vw + dvw, -self.max_angular_speed, self.max_angular_speed)
        
        # Update pose
        self.theta += self.vw * dt
        self.x += self.vx * np.cos(self.theta) * dt
        self.y += self.vx * np.sin(self.theta) * dt

class ObstacleGenerator:
    def __init__(self, world_size=(10, 10), min_obstacles=20, max_obstacles=30, radius=0.5):
        self.world_size = world_size
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.radius = radius
        self.obstacles = []

    def generate(self, safe_zone=[(0,0,0,0)]):
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        self.obstacles = []
        
        for _ in range(num_obstacles):
            while True:
                x = random.uniform(0, self.world_size[0])
                y = random.uniform(0, self.world_size[1])
                
                valid = True
                # Check if obstacle is in safe zone
                for sz in safe_zone:
                    if (sz[0] <= x <= sz[2] and sz[1] <= y <= sz[3]):
                        valid = False
                        break
                if not valid:
                    continue
                self.obstacles.append((x, y))
                break

class SimulationVisualizer:
    def __init__(self):
        # 设置窗口大小
        self.fig = plt.figure(figsize=(24, 8))
        
        # 主仿真视图
        self.ax = self.fig.add_subplot(131)
        
        # Cost矩阵视图
        self.ax_cost = self.fig.add_subplot(132)
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        self.ax.set_aspect('equal')
        
        # 绘制边界框
        # self.ax.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], 'k-')
        
        # 初始化绘图元素
        self.robot_plot, = self.ax.plot([], [], 'bo')
        self.trajectory_plot, = self.ax.plot([], [], 'b-')
        self.target_plot, = self.ax.plot([], [], 'g*', markersize=15)
        self.obstacle_plots = []
        self.trajectory_plots = []
        
        # 速度绘图
        self.ax_vx = self.fig.add_subplot(233)
        self.ax_vx.set_title('Linear Velocity (vx)')
        self.ax_vx.set_xlabel('Time (s)')
        self.ax_vx.set_ylabel('vx (m/s)')
        self.vx_plot, = self.ax_vx.plot([], [], 'b-')
        
        self.ax_vw = self.fig.add_subplot(236)
        self.ax_vw.set_title('Angular Velocity (vw)')
        self.ax_vw.set_xlabel('Time (s)')
        self.ax_vw.set_ylabel('vw (rad/s)')
        self.vw_plot, = self.ax_vw.plot([], [], 'r-')
        
        # 键盘事件处理
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 速度记录
        self.time = []
        self.vx_history = []
        self.vw_history = []
        
        # 仿真控制
        self.SIM_PAUSE_TIME = SIM_PAUSE_TIME  # 默认仿真步长
        self.paused = False

    def update(self, robot, obstacles, trajectory, all_v, all_w, all_trajectories, all_costs, goal):
        # 清除主视图
        self.ax.clear()
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        self.ax.set_aspect('equal')
        
        # 绘制目标点
        self.target_plot, = self.ax.plot(goal[0], goal[1], 'g*', markersize=15)
        
        # 绘制障碍物
        for (x, y) in obstacles:
            circle = plt.Circle((x, y), 0.5, color='r')
            self.ax.add_patch(circle)
            
        # 记录并绘制速度
        if len(self.time) == 0:
            self.time.append(0)
        else:
            self.time.append(self.time[-1] + 0.1)
            
        self.vx_history.append(robot.vx)
        self.vw_history.append(robot.vw)
        
        # 更新速度曲线
        self.vx_plot.set_data(self.time, self.vx_history)
        self.ax_vx.relim()
        self.ax_vx.autoscale_view()
        
        self.vw_plot.set_data(self.time, self.vw_history)
        self.ax_vw.relim()
        self.ax_vw.autoscale_view()

        # 清楚所有轨迹
        for plot in self.trajectory_plots:
            plot.remove()
        self.trajectory_plots = []

        # 绘制预测轨迹
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(all_costs), vmax=max(all_costs))
        
        # 绘制所有预测轨迹
        for v, w, cur_traj, cost in zip(all_v, all_w, all_trajectories, all_costs):
            # 绘制轨迹
            x = [s[0] for s in cur_traj]
            y = [s[1] for s in cur_traj]
            color = cmap(norm(cost))
            plot, = self.ax.plot(x, y, color=color, alpha=0.5)
            self.trajectory_plots.append(plot)
        
        # 绘制机器人
        self.robot_plot, = self.ax.plot(robot.x, robot.y, 'bo')
        
        # 绘制实际轨迹
        self.trajectory_plot, = self.ax.plot([p[0] for p in trajectory],
                                           [p[1] for p in trajectory], 'b-')
        
        self.ax.relim()

        # 绘制cost矩阵
        self.ax_cost.clear()
        v_grid = np.unique(all_v)
        w_grid = np.unique(all_w)
        cost_grid = all_costs.reshape(len(w_grid), len(v_grid))
        v_step = v_grid[1] - v_grid[0]
        w_step = w_grid[1] - w_grid[0]
        self.ax_cost.imshow(cost_grid.T, extent=[min(w_grid)-0.5*w_step, max(w_grid)+0.5*w_step, min(v_grid)-0.5*v_step, max(v_grid)+0.5*v_step],
                          origin='lower', aspect='auto', cmap='viridis')
        # Add text annotations
        for i, w in enumerate(w_grid):
            for j, v in enumerate(v_grid):
                self.ax_cost.text(w, v, f'{cost_grid[i,j]:.1f}', ha='center', va='center', color='black')
        self.ax_cost.set_title('Cost Map')
        self.ax_cost.set_xlabel('Angular Velocity (vw)')
        self.ax_cost.set_ylabel('Linear Velocity (vx)')
        
        # 仿真控制
        if self.SIM_PAUSE_TIME < 0:
            self.paused = True
            while self.paused:
                plt.pause(0.1)
                if plt.fignum_exists(self.fig.number):
                    self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
                else:
                    break
        else:
            plt.pause(self.SIM_PAUSE_TIME)
        
    def motion_model(self, state, v, w):
        """运动学模型"""
        x, y, theta = state
        theta += w * 0.1
        x += v * np.cos(theta) * 0.1
        y += v * np.sin(theta) * 0.1
        return np.array([x, y, theta])
        
    def on_key_press(self, event):
        """键盘事件处理"""
        if event.key == ' ':
            self.paused = False

from dwa import DWAPlanner
def main(COST_FN):
    # Initialize components
    robot = RobotModel()
    obstacle_gen = ObstacleGenerator()
    visualizer = SimulationVisualizer()
    
    # Initialize DWA planner
    dt = 0.1
    planner = DWAPlanner(robot, dt)
    
    # Simulation parameters
    total_time = 20.0
    trajectory = []
    start = (0.0,0.0)  # Start position
    goal = (10.0,10.0)  # Target position
    robot.x = start[0]
    robot.y = start[1]

    # Generate obstacles
    obstacle_gen.generate(safe_zone=[
        (start[0]-1, start[1]-1, start[0]+1, start[1]+1),
        (goal[0]-1, goal[1]-1, goal[0]+1, goal[1]+1)
    ])
    
    arrived = False
    collision = False
    timestamp = 0
    # Main simulation loop
    for t in np.arange(0, total_time, dt):
        # Use DWA to plan velocities
        target_vx, target_vw, all_v, all_w, all_trajectories, all_costs = planner.plan(goal, obstacle_gen.obstacles, COST_FN)
        
        # Update robot state
        robot.update(dt, target_vx, target_vw)
        trajectory.append((robot.x, robot.y))
        
        # Update visualization
        visualizer.update(robot, obstacle_gen.obstacles, trajectory, all_v, all_w, all_trajectories, all_costs, goal)

        # Check if robot has collided with obstacle
        dists = np.linalg.norm(np.array([robot.x, robot.y]) - np.array(obstacle_gen.obstacles), axis=1)
        if np.any(dists < obstacle_gen.radius):
            print(f"Collision detected! time={t}")
            timestamp = t
            collision = True
            break
        # Check if robot has reached goal
        if np.linalg.norm(np.array([robot.x, robot.y]) - np.array(goal)) < 0.2 and \
            np.abs(robot.vx) < 0.2:
            print(f"Goal reached! time={t}")
            timestamp = t
            arrived = True
            break
    if not arrived and not collision:
        print("Robot did not reach goal!..timeout.")
    plt.close()
    return arrived, collision, timestamp

if __name__ == "__main__":
    import os, sys
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
        sys.path.append(app_path)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))
    from cost_fn import SIM_PAUSE_TIME, SEED, cost_fn as COST_FN
    random.seed(SEED)
    print(f"SEED = {SEED}")

    TRY_TIMES = 10
    ARRIVED = 0
    COLLISION = 0
    TIMEOUT = 0
    ARRIVED_TIMES = []
    COLLISION_TIMES = []
    for i in range(TRY_TIMES):
        print(f"Try {i+1}/{TRY_TIMES}......", end="")
        arrived, collision, timestamp = main(COST_FN)
        if arrived:
            ARRIVED += 1
            ARRIVED_TIMES.append(timestamp)
        elif collision:
            COLLISION += 1
            COLLISION_TIMES.append(timestamp)
        else:
            TIMEOUT += 1
    print("-----------------------------\nSimulation finished!")
    print(f"Arrived: {ARRIVED}/{TRY_TIMES}, Collision: {COLLISION}/{TRY_TIMES}, Overtime: {TIMEOUT}/{TRY_TIMES}")
    print(f"Average Arrived times: {np.sum(ARRIVED_TIMES)/ARRIVED if ARRIVED > 0 else 0}")
    print(f"Average Collision times: {np.sum(COLLISION_TIMES)/COLLISION if COLLISION > 0 else 0}")

    # write result to `res2025.gzip`
    import gzip
    import pickle
    with gzip.open("res2025.gzip", "wb") as f:
        pickle.dump((ARRIVED, COLLISION, TIMEOUT, ARRIVED_TIMES, COLLISION_TIMES, SEED), f)

    # # test read
    # with gzip.open("res2025.gzip", "rb") as f:
    #     ARRIVED, COLLISION, OVERTIME, ARRIVED_TIMES, COLLISION_TIMES, SEED = pickle.load(f)
    # print(f"Arrived: {ARRIVED}/{TRY_TIMES}, Collision: {COLLISION}/{TRY_TIMES}, Overtime: {TIMEOUT}/{TRY_TIMES}")
    # print(f"Average Arrived times: {np.sum(ARRIVED_TIMES)/ARRIVED}")
    # print(f"Average Collision times: {np.sum(COLLISION_TIMES)/COLLISION}")
    # print(f"SEED: {SEED}")
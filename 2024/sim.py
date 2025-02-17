import numpy as np

class Sim:
    robot_max_speed = 1.0 # m/s
    observation_range = 0.5 # m
    done_range = 0.1 # m

    def __init__(self,N=10):
        self.reset(N)

    def reset(self, N):
        self.t = 0.0
        self.robot = np.array([0,0])
        self.target = np.array([0,0])
        # random N point in 1m circle
        self.targets = np.random.rand(N,2)*2-1
        # scale to 1m circle
        self.targets = self.targets / np.clip(np.linalg.norm(self.targets, axis=1).reshape(-1,1),1.0,2.0)

        # mark all targets as undone
        self.targets_done = np.zeros(N, dtype=bool)

        self.current_debug = ""

    def done(self):
        return np.all(self.targets_done)

    def obs(self):
        # get all undone targets
        undone_targets = self.targets[~self.targets_done]
        # only dist < self.observation_range is visible
        return undone_targets[np.linalg.norm(undone_targets - self.robot, axis=1) < self.observation_range]

    def step(self, dt):
        # limit the target to the intersection of robot2target line and circle wall

        dist = np.linalg.norm(np.array(self.target) - np.array(self.robot))
        if dist < self.robot_max_speed*dt:
            self.robot = self.target
        else:
            self.robot = self.robot + self.robot_max_speed*dt*(self.target - self.robot)/dist
        self.t += dt
        # if robot to target dist < self.done_range, then target is done
        self.targets_done[np.linalg.norm(self.targets - self.robot, axis=1) < self.done_range] = True
        # return robot_pos, visiable_targets, all_targets[done==true]
        return self.robot, self.obs(), self.targets[self.targets_done], self.targets, self.t, self.current_debug

    def set_target(self, *target):
        self.target = np.array([*target])

    def set_target_as_closest(self):
        # get all undone targets
        undone_targets = self.obs()
        # check undone_targets is empty
        if undone_targets.size == 0:
            self.target = self.robot
            return
        # get the closest undone target
        self.target = undone_targets[np.argmin(np.linalg.norm(undone_targets - self.robot, axis=1))]

    def get_robot(self):
        return self.robot
    
    def check(self, condition, value=None):
        if condition == 'have_targets':
            return self.obs().size > 0
        else:
            return self._check_2(condition, value)
    
    def _check_2(self, condition, value=None):
        # check if value is a number
        if value is None:
            raise ValueError('Argument is None or Condition unknown : {}'.format(condition))

        if condition == 'x_larger_than':
            return self.robot[0] > value
        elif condition == 'y_larger_than':
            return self.robot[1] > value
        elif condition == 'x_smaller_than':
            return self.robot[0] < value
        elif condition == 'y_smaller_than':
            return self.robot[1] < value
        else:
            raise ValueError('Unknown condition: {}'.format(condition))
    
    def exe(self, action, *value):
        if action == 'to_point':
            self.set_target(*value)
        elif action == 'to_closest_target':
            self.set_target_as_closest()
        else:
            raise ValueError('Unknown action: {}'.format(action))

    def decision(self, tree):
        self.current_debug = ""
        for node in tree:
            if node['Type'] == 'Judgement':
                if self.check(*node['Arguments']):
                    return self.decision(node['Children'])
            elif node['Type'] == 'Action':
                self.exe(*node['Arguments'])
                self.current_debug = node['ID'] + ": " + str(node['Arguments'])
                return
# use plt
import matplotlib.pyplot as plt
class Visualizer:
    def __init__(self):
        # set window square
        plt.rcParams["figure.figsize"] = (10,10)
        plt.ion()
        plt.show()
        pass
    def update(self, ax, robot, obs, targets_done, targets, t, debug_str):
        ax.clear()
        ax.plot(targets[:,0], targets[:,1], 'rx')
        ax.plot(robot[0], robot[1], color='b', marker='P', markersize=20)
        # draw robot vis circle 0.1m
        circle = plt.Circle((robot[0], robot[1]), Sim.done_range, color='r', fill=False)
        ax.add_artist(circle)
        circle = plt.Circle((robot[0], robot[1]), Sim.observation_range, color='g', fill=False)
        ax.add_artist(circle)
        ax.plot(obs[:,0], obs[:,1], 'bo')
        ax.plot(targets_done[:,0], targets_done[:,1], 'go')
        # draw walls
        circle = plt.Circle((0,0), 1, color='black', fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        # set x text to show time
        ax.text(-1.0, -0.9, 't={:.2f}'.format(t), fontsize=12)
        ax.text(-1.0, -1.0, debug_str, fontsize=12)
        plt.pause(0.01)

if __name__ == '__main__':
    import time, sys, os
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
        sys.path.append(app_path)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))
    sim = Sim()
    vis = Visualizer()
    target = np.array([1,0])
    
    # set window square
    fig, ax = plt.subplots()

    import strategy as S
    print(S.DECISION_TREE)

    config = S.CONFIG
    config.update({
        "repeat_times" : 10,
        "max_time" : 20, # seconds
        "fps" : 20, # frames per second
    })
    dt = 1.0/config["fps"]
    np.random.seed(config["seed"])

    results = []
    for times in range(config["repeat_times"]):
        sim.reset(10)
        for i in range(config['max_time']*config['fps']):
            sim.decision(S.DECISION_TREE)
            vis.update(ax,*sim.step(dt))
            time.sleep(dt/config['speed_up'])
            if sim.done():
                break
        done_count = np.sum(sim.targets_done)
        results.append([sim.t,done_count])
    print("Results : ")
    for r in results:
        print("Time: {:.2f}, Done: {}".format(r[0],r[1]))
    
    import gzip, struct
    with gzip.open(os.path.join(app_path,"results-{}-{}.gz".format(config["seed"],config['ID'])), "wb") as f:
        f.write(b"PYSSL_RESULT")
        f.write(struct.pack('>QsI',config["seed"],str(config['ID']).encode(),len(results)))
        for r in results:
            f.write(struct.pack('>fi',r[0],r[1]))
    # for i in range(100):
    #     sim.decision(S.DECISION_TREE)
    #     vis.update(ax,*sim.step(dt))
    #     time.sleep(dt)
    #     if sim.done():
    #         break

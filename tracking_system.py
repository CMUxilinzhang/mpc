# src/tracking_system.py

import numpy as np

class TrackingSystem:
    def __init__(self, dt=0.1):
        self.dt = dt  # 时间步长
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0.0, 0.0, 0.0])
        self.agent_vel = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.random.uniform(-10, 10, size=(3,))
        self.target_vel = np.random.uniform(-1, 1, size=(3,))
        self.time_step = 0

    def step(self, agent_acc):
        """
        执行一步更新

        参数:
            agent_acc (numpy array): 代理的加速度 [ax, ay, az]

        返回:
            state (dict): 更新后的状态
            reward (float): 奖励
            done (bool): 是否终止
        """
        # 更新代理状态
        self.agent_vel += agent_acc * self.dt
        self.agent_pos += self.agent_vel * self.dt

        # 更新目标状态（假设目标以恒定速度运动）
        self.target_pos += self.target_vel * self.dt

        # 计算距离
        distance = np.linalg.norm(self.agent_pos - self.target_pos)

        # 奖励函数
        reward = -distance  # 距离越小，奖励越大

        # 定义终止条件
        reach_threshold = 0.5
        max_steps = 500
        self.time_step += 1
        #done = distance < reach_threshold or self.time_step >= max_steps
        done = self.time_step >= max_steps

        return {
            'agent_pos': self.agent_pos.copy(),
            'agent_vel': self.agent_vel.copy(),
            'target_pos': self.target_pos.copy(),
            'target_vel': self.target_vel.copy()
        }, reward, done

# src/mpc_controller.py

import cvxpy as cp
import numpy as np

class MPCController:
    def __init__(self, system, horizon=10, weight_error=1.0, weight_control=0.1, max_acc=1.0):
        """
        初始化MPC控制器

        参数:
            system (TrackingSystem): 系统模型
            horizon (int): 预测步长
            weight_error (float): 追踪误差权重
            weight_control (float): 控制输入权重
            max_acc (float): 最大加速度
        """
        self.system = system
        self.horizon = horizon
        self.weight_error = weight_error
        self.weight_control = weight_control
        self.max_acc = max_acc

        # 定义优化变量
        self.acc_vars = cp.Variable((horizon, 3))  # 每步的加速度
        self.agent_pos = cp.Variable((horizon + 1, 3))
        self.agent_vel = cp.Variable((horizon + 1, 3))

    def solve(self):
        """
        解决优化问题并返回第一个控制输入

        返回:
            first_acc (numpy array): 第一个加速度控制输入 [ax, ay, az]
        """
        constraints = []
        cost = 0

        # 更新目标位置序列基于当前目标的位置和速度
        target_pos_seq = [self.system.target_pos + (t + 1) * self.system.target_vel * self.system.dt for t in range(self.horizon)]

        # 初始条件
        constraints += [
            self.agent_pos[0] == self.system.agent_pos,
            self.agent_vel[0] == self.system.agent_vel
        ]

        for t in range(self.horizon):
            # 动力学约束
            constraints += [
                self.agent_vel[t + 1] == self.agent_vel[t] + self.acc_vars[t] * self.system.dt,
                self.agent_pos[t + 1] == self.agent_pos[t] + self.agent_vel[t] * self.system.dt
            ]

            # 加速度约束
            constraints += [
                cp.abs(self.acc_vars[t]) <= self.max_acc
            ]

            # 目标追踪误差
            error = self.agent_pos[t + 1] - target_pos_seq[t]
            cost += self.weight_error * cp.sum_squares(error)

            # 控制输入变化惩罚
            cost += self.weight_control * cp.sum_squares(self.acc_vars[t])

        # 定义优化问题
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # 求解优化问题
        prob.solve(solver=cp.OSQP)

        if prob.status != cp.OPTIMAL:
            print("MPC optimization problem did not solve to optimality.")
            return np.zeros(3)

        # 返回第一个加速度控制输入
        first_acc = self.acc_vars.value[0]
        return first_acc

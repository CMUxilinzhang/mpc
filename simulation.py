# src/simulation.py

import numpy as np
from tracking_system import TrackingSystem
from mpc_controller import MPCController
from visualize import visualize_dynamic, visualize_realtime

def run_simulation(total_steps=500, realtime=False):
    """
    运行MPC追踪仿真，并进行可视化

    参数:
        total_steps (int): 最大仿真步数
        realtime (bool): 是否使用实时可视化
    """
    # 初始化系统和控制器
    system = TrackingSystem(dt=0.1)
    mpc = MPCController(system, horizon=10, weight_error=1.0, weight_control=0.1, max_acc=1.0)

    # 初始化数据记录
    agent_traj = [system.agent_pos.copy()]
    target_traj = [system.target_pos.copy()]

    # 重置系统
    system.reset()

    for step in range(total_steps):
        # 解决MPC优化问题，获取控制输入
        acc = mpc.solve()

        # 执行一步更新
        state, reward, done = system.step(acc)

        # 记录轨迹
        agent_traj.append(system.agent_pos.copy())
        target_traj.append(system.target_pos.copy())

        if done:
            print(f"目标已达到或达到最大步数: {step + 1}")
            break

    # 转换为numpy数组
    agent_traj = np.array(agent_traj)
    target_traj = np.array(target_traj)

    # 动态可视化轨迹
    if realtime:
        visualize_realtime(agent_traj, target_traj, total_steps=len(agent_traj))
    else:
        visualize_dynamic(agent_traj, target_traj)

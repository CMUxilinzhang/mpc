# src/visualize.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import time

def visualize_dynamic(agent_traj, target_traj):
    """
    动态可视化代理和目标的三维轨迹

    参数:
        agent_traj (numpy array): 代理轨迹 [N, 3]
        target_traj (numpy array): 目标轨迹 [N, 3]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围
    all_positions = np.vstack((agent_traj, target_traj))
    max_range = np.array([all_positions[:,0].max()-all_positions[:,0].min(), 
                          all_positions[:,1].max()-all_positions[:,1].min(), 
                          all_positions[:,2].max()-all_positions[:,2].min()]).max() / 2.0

    mid_x = (all_positions[:,0].max()+all_positions[:,0].min()) * 0.5
    mid_y = (all_positions[:,1].max()+all_positions[:,1].min()) * 0.5
    mid_z = (all_positions[:,2].max()+all_positions[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Tracking Trajectories - Dynamic Visualization')

    # 初始化线条和点
    agent_line, = ax.plot([], [], [], label='Agent Trajectory', color='blue')
    target_line, = ax.plot([], [], [], label='Target Trajectory', color='red')

    agent_point, = ax.plot([], [], [], marker='o', color='blue', markersize=5, label='Agent')
    target_point, = ax.plot([], [], [], marker='o', color='red', markersize=5, label='Target')

    distance_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    ax.legend()

    # 初始化函数
    def init():
        agent_line.set_data([], [])
        agent_line.set_3d_properties([])

        target_line.set_data([], [])
        target_line.set_3d_properties([])

        agent_point.set_data([], [])
        agent_point.set_3d_properties([])

        target_point.set_data([], [])
        target_point.set_3d_properties([])

        distance_text.set_text("")

        return agent_line, target_line, agent_point, target_point, distance_text

    # 更新函数
    def update(num, agent_traj, target_traj, agent_line, target_line, agent_point, target_point, distance_text):
        # 更新轨迹线
        agent_line.set_data(agent_traj[:num,0], agent_traj[:num,1])
        agent_line.set_3d_properties(agent_traj[:num,2])

        target_line.set_data(target_traj[:num,0], target_traj[:num,1])
        target_line.set_3d_properties(target_traj[:num,2])

        # 更新当前点
        agent_point.set_data(agent_traj[num-1,0], agent_traj[num-1,1])
        agent_point.set_3d_properties(agent_traj[num-1,2])

        target_point.set_data(target_traj[num-1,0], target_traj[num-1,1])
        target_point.set_3d_properties(target_traj[num-1,2])

        # 更新距离文本
        distance = np.linalg.norm(agent_traj[num-1] - target_traj[num-1])
        distance_text.set_text(f"Step: {num}\nDistance: {distance:.2f}")

        return agent_line, target_line, agent_point, target_point, distance_text

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(agent_traj),
                                  fargs=(agent_traj, target_traj, agent_line, target_line, 
                                         agent_point, target_point, distance_text),
                                  init_func=init, blit=False, interval=50, repeat=False)

    plt.show()

def visualize_realtime(agent_traj, target_traj, total_steps):
    """
    实时动态可视化代理和目标的三维轨迹

    参数:
        agent_traj (list of numpy arrays): 代理轨迹
        target_traj (list of numpy arrays): 目标轨迹
        total_steps (int): 总步数
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围
    all_positions = np.vstack((agent_traj, target_traj))
    max_range = np.array([all_positions[:,0].max()-all_positions[:,0].min(), 
                          all_positions[:,1].max()-all_positions[:,1].min(), 
                          all_positions[:,2].max()-all_positions[:,2].min()]).max() / 2.0

    mid_x = (all_positions[:,0].max()+all_positions[:,0].min()) * 0.5
    mid_y = (all_positions[:,1].max()+all_positions[:,1].min()) * 0.5
    mid_z = (all_positions[:,2].max()+all_positions[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Tracking Trajectories - Real-time Visualization')

    # 初始化线条和点
    agent_line, = ax.plot([], [], [], label='Agent Trajectory', color='blue')
    target_line, = ax.plot([], [], [], label='Target Trajectory', color='red')

    agent_point, = ax.plot([], [], [], marker='o', color='blue', markersize=5, label='Agent')
    target_point, = ax.plot([], [], [], marker='o', color='red', markersize=5, label='Target')

    distance_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    ax.legend()

    plt.ion()
    plt.show()

    for i in range(1, total_steps + 1):
        if i > len(agent_traj):
            break

        agent_current = agent_traj[:i]
        target_current = target_traj[:i]

        # 更新轨迹线
        agent_line.set_data(agent_current[:,0], agent_current[:,1])
        agent_line.set_3d_properties(agent_current[:,2])

        target_line.set_data(target_current[:,0], target_current[:,1])
        target_line.set_3d_properties(target_current[:,2])

        # 更新当前点
        agent_point.set_data(agent_traj[i-1,0], agent_traj[i-1,1])
        agent_point.set_3d_properties(agent_traj[i-1,2])

        target_point.set_data(target_traj[i-1,0], target_traj[i-1,1])
        target_point.set_3d_properties(target_traj[i-1,2])

        # 更新距离文本
        distance = np.linalg.norm(agent_traj[i-1] - target_traj[i-1])
        distance_text.set_text(f"Step: {i}\nDistance: {distance:.2f}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)  # 控制动画速度

    plt.ioff()
    plt.show()

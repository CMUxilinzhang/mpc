# scripts/run_simulation.py

import argparse
from simulation import run_simulation

def parse_args():
    parser = argparse.ArgumentParser(description="MPC-Based 3D Tracking Simulation")
    parser.add_argument('--steps', type=int, default=500, help='Total simulation steps')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time visualization')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_simulation(total_steps=args.steps, realtime=args.realtime)


import json
import numpy as np

RESULTS_PATH = "frontend/public/experiment_results.json"

def main():
    try:
        with open(RESULTS_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    print(f"{'Experiment':<25} | {'Metric':<20} | {'RL':<10} | {'ES':<10}")
    print("-" * 75)
    
    # helper
    def get(agent, exp, metric):
        try:
            val = data[agent][exp]["metrics"][metric]
            return f"{val:.3f}"
        except:
            return "N/A"

    # Exp 1
    print(f"{'1. New World':<25} | {'Efficiency':<20} | {get('RL', 'exp_1', 'efficiency'):<10} | {get('ES', 'exp_1', 'efficiency'):<10}")
    print(f"{'':<25} | {'Survival':<20} | {get('RL', 'exp_1', 'survival'):<10} | {get('ES', 'exp_1', 'survival'):<10}")
    
    # Exp 2
    print(f"{'2. Ice Patch':<25} | {'Max Deviation':<20} | {get('RL', 'exp_2', 'max_deviation'):<10} | {get('ES', 'exp_2', 'max_deviation'):<10}")
    print(f"{'':<25} | {'Recovery Rate':<20} | {get('RL', 'exp_2', 'recovery_rate'):<10} | {get('ES', 'exp_2', 'recovery_rate'):<10}")
    
    # Exp 3
    print(f"{'3. Foggy (Head)':<25} | {'Survival':<20} | {get('RL', 'exp_3', 'heading_survival'):<10} | {get('ES', 'exp_3', 'heading_survival'):<10}")
    print(f"{'':<25} | {'Wobble':<20} | {get('RL', 'exp_3', 'heading_wobble'):<10} | {get('ES', 'exp_3', 'heading_wobble'):<10}")
    print(f"{'3. Foggy (Lat)':<25} | {'Survival':<20} | {get('RL', 'exp_3', 'lateral_survival'):<10} | {get('ES', 'exp_3', 'lateral_survival'):<10}")
    
    # Exp 4
    print(f"{'4. Blindfold':<25} | {'Survival':<20} | {get('RL', 'exp_4', 'survival'):<10} | {get('ES', 'exp_4', 'survival'):<10}")
    print(f"{'':<25} | {'Avg Speed':<20} | {get('RL', 'exp_4', 'avg_speed'):<10} | {get('ES', 'exp_4', 'avg_speed'):<10}")
    
    # Exp 5
    print(f"{'5. Wake (Wall)':<25} | {'Success':<20} | {get('RL', 'exp_5', 'wall_facer_success'):<10} | {get('ES', 'exp_5', 'wall_facer_success'):<10}")
    print(f"{'5. Wake (Mid)':<25} | {'Success':<20} | {get('RL', 'exp_5', 'mid_turn_success'):<10} | {get('ES', 'exp_5', 'mid_turn_success'):<10}")

if __name__ == "__main__":
    main()

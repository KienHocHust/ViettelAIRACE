import matlab.engine
import numpy as np
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import csv 


ENG = matlab.engine.start_matlab()

def _serialize_array(value):
    if isinstance(value, (list, np.ndarray)):
        return ",".join([f"{float(v):.4f}" for v in np.ravel(value)])
    return value

def log_metrics(metrics, state=None, action=None, log_file="logs/default_log.csv"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    METRIC_NAMES = [
        "current_energy",
        "currentconnected_ues",
        "currentdrop_rate",
        "currentlatency",
        "drop_penalty",
        "latency_penalty",
        "reward"
    ]
    
    file_exists = os.path.isfile(log_file)
    if not file_exists:
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "episode",
                "step",
                "actual_reward",
                "state",
                "action"
            ] + METRIC_NAMES
            writer.writerow(header)

    state_str = _serialize_array(state)
    action_str = _serialize_array(action)
    
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        row = [
            metrics.get("episode", 0),
            metrics.get("step", 0),
            float(metrics.get("reward", 0)),
            state_str,
            action_str
        ] + [metrics.get(k, None) for k in METRIC_NAMES]
        writer.writerow(row)


class MatlabSimEnv(gym.Env):
    
    def __init__(self, matlab_code_path, scenario_path):
        print(f"Đang khởi tạo môi trường với code MATLAB tại: {matlab_code_path}")

        self.scenario_path = scenario_path
        
        scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
        self.log_file_path = f"logs/training_log_{scenario_name}.csv"
        
        self.eng = ENG
        
        # path 
        simcore_path = os.path.join(matlab_code_path, 'simCore')
        utils_path = os.path.join(matlab_code_path, 'utils')        
        self.eng.addpath(matlab_code_path, nargout=0)  
        self.eng.addpath(simcore_path, nargout=0) 
        self.eng.addpath(utils_path, nargout=0)  
                        
        state_matlab, n_cells, max_steps, n_ues = self.eng.reset_env(
            self.scenario_path, 
            nargout=4
        )
        
        initial_state = np.array(state_matlab[0], dtype=np.float32) 
        self.n_cells = int(n_cells)
        self.max_steps = int(max_steps)
        self.n_ues = int(n_ues)      
        
        self.state_dim = initial_state.shape[1] 
        self.action_dim = self.n_cells
        
        self.action_space = spaces.Box(low=0.0, high=1.0, 
                                       shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.state_dim,), dtype=np.float32)
        
        self.current_step = 0
    
        print("\n" + "="*40)
        print("MÔI TRƯỜNG ĐÃ KHỞI TẠO THÀNH CÔNG")
        print(f"  - Kịch bản: {self.scenario_path}")
        print(f"  - Số lượng Cells (action_dim): {self.action_dim}")
        print(f"  - Số lượng UEs: {self.n_ues}")
        print(f"  - Max step: {self.max_steps}")
        print(f"  - Kích thước State: {self.state_dim}")
        print("="*40 + "\n")


    def step(self, action):
        self.current_step += 1
        matlab_action = matlab.double(action.tolist())
        next_state_matlab, reward_matlab, done, metrics_matlab = self.eng.step_env(matlab_action, nargout=4)
        
        truncated = False
        terminated = bool(done)
        
        if self.current_step >= self.max_steps:
            truncated = True
            terminated = True
            
        next_state = np.array(next_state_matlab[0], dtype=np.float32)
        reward = float(reward_matlab)
        metrics_dict = {k: metrics_matlab[k] for k in metrics_matlab.keys()}
        
        log_metrics(metrics_dict, state=next_state, action=action, log_file=self.log_file_path)
        
        return next_state.flatten(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        state_matlab, _, _, _ = self.eng.reset_env(self.scenario_path, nargout=4)      
        initial_state = np.array(state_matlab[0], dtype=np.float32)
        return initial_state.flatten(), {}
    
    def print_final_summary(self):
            self.eng.print_final_summary(nargout=0)


    def close(self):
            self.eng.quit()
      

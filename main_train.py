import os
import time
import numpy as np
import random
import torch
import glob 
from datetime import datetime
from gym_wrapper import MatlabSimEnv 
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCENARIO_JSON = 'high_speed'
MATLAB_CODE_PATH = os.path.abspath('simulation')
MAX_EPISODES = 100
MODEL_SAVE_PREFIX = 'ppo_model'
STATS_PATH = "vec_normalize_stats.pkl"

def make_env():
    env = MatlabSimEnv(
        scenario_path=SCENARIO_JSON,
        matlab_code_path=MATLAB_CODE_PATH
    )
    return env

if __name__ == '__main__':
    env_raw = DummyVecEnv([make_env])
    
    if os.path.exists(STATS_PATH):
        print(f"Đang tải stats chuẩn hóa từ: {STATS_PATH}")
        env = VecNormalize.load(STATS_PATH, env_raw)
        env.training = True
        env.norm_reward = True
    else:
        print("Không tìm thấy stats, tạo VecNormalize mới và chuẩn hóa.")
        env = VecNormalize(env_raw, norm_obs=True, norm_reward=True, clip_obs=10.)

    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    list_of_models = glob.glob(f'{MODEL_SAVE_PREFIX}_*.zip')
    if list_of_models:
        latest_model_file = max(list_of_models, key=os.path.getctime)
        print(f"Đã tìm thấy model cũ: {latest_model_file}")
        model = PPO.load(latest_model_file, env=env)
        new_logger = configure(log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        print("Không tìm thấy model cũ, huấn luyện từ đầu.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    try:
        total_timesteps = env.envs[0].max_steps * MAX_EPISODES
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_MatlabSim")
    except KeyboardInterrupt:
        pass
    finally:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{MODEL_SAVE_PREFIX}_{timestamp}.zip"
        model.save(save_path)
        env.save(STATS_PATH)
        
        print(f"\n\n------- HUẤN LUYỆN KẾT THÚC ---------")
        print(f"Model đã được lưu tại: {save_path}")
        print(f"Stats chuẩn hóa đã được lưu tại: {STATS_PATH}")
        
        try:
            env.envs[0].print_final_summary()
        except AttributeError:
            pass
        
        env.close()

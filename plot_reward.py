import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE_PATH = 'logs/training_log_high_speed.csv'
OUTPUT_IMAGE_NAME = 'reward_chart_smoothed.png'
ROLLING_WINDOW = 1000
START_STEP = 1

def plot_rewards():
    
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Không tìm thấy file!!!!")
        return

    try:
        data = pd.read_csv(LOG_FILE_PATH)
        
        if data.empty:
            print("File log trống!!!!")
            return
        
        step_rewards_smooth = data['actual_reward'].rolling(window=ROLLING_WINDOW).mean()
        
        filtered_rewards_smooth = step_rewards_smooth.loc[step_rewards_smooth.index >= START_STEP]

        if filtered_rewards_smooth.empty:
             print(f"Không có dữ liệu reward từ step {START_STEP}")
             return

        fig, ax = plt.subplots(1, 1, figsize=(12, 6)) 
        
        ax.plot(filtered_rewards_smooth, label=f'Rolling Avg (Window={ROLLING_WINDOW})', color='orange')
        
        ax.set_title(f'Reward theo từng Step (Từ Step {START_STEP})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Smoothed Reward')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE_NAME)
        
        print(f"\nĐã lưu biểu đồ tại: {OUTPUT_IMAGE_NAME}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    plot_rewards()
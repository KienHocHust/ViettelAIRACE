import torch

path = r"C:\Users\TGDD\Documents\MATLAB\SourceCode\ppo_model_220425.pth"

# Load checkpoint
checkpoint = torch.load(path, map_location="cpu")

# In danh sách key có trong checkpoint
print("Các trường có trong checkpoint:")
for k in checkpoint.keys():
    print("-", k)

# Nếu muốn xem chi tiết
print("\nTổng số episode:", checkpoint.get("total_episodes", "N/A"))
print("Tổng số step:", checkpoint.get("total_steps", "N/A"))

# Nếu muốn xem cấu trúc mạng (chỉ in kích thước tensor)
print("\nActor layers:")
for name, param in checkpoint['actor_state_dict'].items():
    print(f"{name}: {tuple(param.shape)}")

print("\nCritic layers:")
for name, param in checkpoint['critic_state_dict'].items():
    print(f"{name}: {tuple(param.shape)}")

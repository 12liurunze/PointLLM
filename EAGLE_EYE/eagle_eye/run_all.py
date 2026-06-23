import subprocess

# 定义两个脚本路径
script1 = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/ge_data/get_data_vedio_all_qwen2.5vl.py"
script2 = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/train/train_qwenvl2.5_video.py"

# 运行第一个脚本
print(f"Running {script1}...")
result1 = subprocess.run(["python", script1], check=True)
print(f"Finished {script1}.")

# 运行第二个脚本
print(f"Running {script2}...")
result2 = subprocess.run(["python", script2], check=True)
print(f"Finished {script2}.")

import subprocess
import torch

def launch_training():
    # 自动检测GPU数量
    num_gpus = torch.cuda.device_count()

    command = [
        "accelerate", "launch",
        "--mixed_precision", "fp16",
        "--gpu_ids", "all",
        "--num_processes", str(num_gpus),
        "tools/train.py",
        "+exp=224x400",
        "runner=8gpus"
    ]
    
    try:
        # 启动命令
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    launch_training()

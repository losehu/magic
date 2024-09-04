import subprocess
import torch
import shutil
import os

def launch_accelerate():
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Define the directory to be deleted
    img_gen_dir = './tmp'

    # Check if the directory exists, and if so, delete it
    if os.path.exists(img_gen_dir):
        shutil.rmtree(img_gen_dir)
        print(f"Deleted directory: {img_gen_dir}")
    else:
        print(f"Directory does not exist: {img_gen_dir}")

    # Prepare the command
    command = [
        "accelerate", "launch",
        "--mixed_precision", "fp16",
        "--gpu_ids", "all",
        "--num_processes", str(num_gpus),  # Use the number of GPUs
        "perception/data_prepare/val_set_gen.py",
        "resume_from_checkpoint=/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-29_17-02_224x400/weight-E125-S72975",
        "task_id=224x400",
        "fid.img_gen_dir=" + "tmp/224x400/samples",
        "+fid=data_gen",
        "+exp=224x400"
    ]
    
    # Execute the command as a subprocess
    result = subprocess.run(command, check=True)

    # Optional: Check the result
    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print(f"Command failed with return code: {result.returncode}")

if __name__ == "__main__":
    launch_accelerate()

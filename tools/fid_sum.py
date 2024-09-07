import subprocess
import torch
import shutil
import os
import re
weight_path="/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E34-S20575"
match = str(re.search(r'(\d+)$', weight_path).group(1))
tmp_dir='tmp-'+match+"/224x400/samples"
def launch_accelerate():
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # # Define the directory to be deleted
    # img_gen_dir = './tmp-'+match

    # # Check if the directory exists, and if so, delete it
    # if os.path.exists(img_gen_dir):
    #     shutil.rmtree(img_gen_dir)
    #     print(f"Deleted directory: {img_gen_dir}")
    # else:
    #     print(f"Directory does not exist: {img_gen_dir}")

    # Prepare the command
    command = [
        "accelerate", "launch",
        "--mixed_precision", "fp16",
        "--gpu_ids", "all",
        "--num_processes", str(num_gpus),  # Use the number of GPUs
        "perception/data_prepare/val_set_gen.py",
        "resume_from_checkpoint="+weight_path,
        "task_id=224x400",
        "fid.img_gen_dir=" + tmp_dir,
        "+fid=data_gen",
        "+exp=224x400"
    ]
    
    # Execute the command as a subprocess
    subprocess.run(command, check=True)
    command = [
        'python', 'tools/fid_score.py', 'cfg',
        'resume_from_checkpoint='+weight_path,
        'fid.rootb='+tmp_dir
    ]
    

    subprocess.run(command, check=True)
 

if __name__ == "__main__":
    launch_accelerate()

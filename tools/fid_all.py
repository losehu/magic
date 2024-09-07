import subprocess
import torch
import shutil
import os
import re
weight_paths=[
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E34-S20575",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E38-S22875",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E42-S25175",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E46-S27475",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E51-S30350",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E55-S32650",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E60-S35525",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E64-S37825",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E68-S40125",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-29_10-47_224x400/weight-E86-S50575",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-29_17-02_224x400/weight-E103-S60325",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-29_17-02_224x400/weight-E120-S70100",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-09-04_19-29_224x400/weight-E135-S78300",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-09-04_19-29_224x400/weight-E165-S95550",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-09-04_19-29_224x400/weight-E180-S104175",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-09-06_13-21_224x400/weight-E195-S113175",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-09-06_13-21_224x400/weight-E210-S121800",
]
def launch_accelerate():
    for weight_path in weight_paths:
        match = str(re.search(r'(\d+)$', weight_path).group(1))
        tmp_dir='tmp-'+match+"/224x400/samples"
        if os.path.exists(tmp_dir):
            print(f'{tmp_dir} not empty!!')
            continue
        if not os.path.exists(weight_path+'/hydra'):
            shutil.copytree(weight_path+"/../hydra",weight_path+'/hydra' , dirs_exist_ok=True)

        num_gpus = torch.cuda.device_count()
        command1 = [
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
        subprocess.run(command1, check=True)
        command2 = [
            'python', 'tools/fid_score.py', 'cfg',
            'resume_from_checkpoint='+weight_path,
            'fid.rootb='+tmp_dir
        ]
        subprocess.run(command2, check=True)
 

if __name__ == "__main__":
    launch_accelerate()

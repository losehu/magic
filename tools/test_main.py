import torch
import subprocess
from multiprocessing import Process

def run_command(gpu_id):
    command = [
        "python", "perception/data_prepare/val_set_gen.py",
        "+exp=224x400", "task_id=224x400",
        "resume_from_checkpoint=./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400",
        "fid.img_gen_dir=./tmp/224x400", "+fid=data_gen",
        "runner.pipeline_param.use_zero_map_as_unconditional=true"
    ]
    
    # 设置CUDA_VISIBLE_DEVICES环境变量来指定使用哪个GPU
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    
    # 运行命令
    subprocess.run(command, env=env)

def main():
    # 获取可用GPU的数量
    num_gpus = torch.cuda.device_count()
    
    processes = []
    
    # 为每个GPU启动一个进程
    for gpu_id in range(num_gpus):
        p = Process(target=run_command, args=(gpu_id,))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

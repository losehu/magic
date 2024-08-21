
import sys
import subprocess
import shutil
import os
def main():
    # 定义命令和参数
    command = [
        "python", "perception/data_prepare/val_set_gen.py",
        "resume_from_checkpoint=./magicdrive-log/SDv1.5mv-rawbox_2024-08-17_00-43_224x400_1_CAM_FRONT",
        "task_id=224x400",
        "fid.img_gen_dir=./tmp/224x400",
        "+fid=data_gen",
        "+exp=224x400"
    ]
    folder_path = './tmp'
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 已成功删除。")
    else:
        print(f"文件夹 '{folder_path}' 不存在。")
    subprocess.run(command)


if __name__ == "__main__":
    main()


import subprocess

command = [
    "python", "tools/fid_score.py", "cfg",
    "resume_from_checkpoint=./magicdrive-log/SDv1.5mv-rawbox_2024-08-17_00-43_224x400_1_CAM_FRONT",
    "fid.rootb=tmp/224x400"
]

# 执行命令
subprocess.run(command)

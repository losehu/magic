import subprocess
import torch
import shutil
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
weight_paths=[
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E34-S20575",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E35-S21150",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E36-S21725",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E37-S22300",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E38-S22875",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E39-S23450",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E40-S24025",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E41-S24600",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E42-S25175",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E43-S25750",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E44-S26325",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E45-S26900",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E46-S27475",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E47-S28050",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E48-S28625",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E49-S29200",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E50-S29775",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E51-S30350",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E52-S30925",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E53-S31500",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E54-S32075",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E55-S32650",
"/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-27_23-10_224x400/weight-E58-S34375",
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


# 读取数据
def read_data(file_path):
    x = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
    return x, y


def plot_data(x, y, output_file):
    # 对 x 和 y 数据根据 x 的值进行排序
    sorted_data = sorted(zip(x, y), key=lambda pair: pair[0])
    x_sorted, y_sorted = zip(*sorted_data)  # 解压成两个列表

    plt.figure(figsize=(12, 8))
    plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', color='b', label='Data')
    plt.xlabel('STEP')
    plt.ylabel('FID')
    plt.title('FID')
    plt.legend()
    plt.grid(True)

    # 计算x轴的范围
    x_min, x_max = min(x_sorted), max(x_sorted)
    # 扩展x轴范围以减少标签重叠
    x_padding = (x_max - x_min) * 0.1
    plt.xlim(x_min - x_padding, x_max + x_padding)

    plt.savefig(output_file)  # 保存图形到文件
    plt.close()  # 关闭图形，以便释放资源


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
    #绘制FID图像
    file_path = 'fid.txt'  # 替换为您的文件路径
    output_file = 'fid.png'  # 设置保存图形的文件名
    x, y = read_data(file_path)
    plot_data(x, y, output_file)
    #将图片统一输出
    img_names=[
    "1531281439800013_gen_0.jpg",
    "1538985054947711_gen_0.jpg",
    "1542799485148043_gen_0.jpg",
    ]
    if not os.path.exists('./tmp_sum'):
        os.makedirs('./tmp_sum')
    for img_name in img_names:
        directory_path='./tmp_sum/'+str(re.match(r'^(\d+)_gen_', img_name).group(1))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for weight_path in weight_paths:
            match = str(re.search(r'(\d+)$', weight_path).group(1))
            img_dir='tmp-'+match+"/224x400/samples"+"/CAM_PANO/"+img_name
            if os.path.isfile(img_dir):
                # if not os.path.isfile(directory_path+"/"+img_name)
                shutil.copy2(img_dir, directory_path+"/"+match+'.png')
                input_image_path = img_dir 
                output_image_path = directory_path+"/"+match+'.png' 
                with Image.open(input_image_path) as img:
                    img_resized = img.resize((2400, 160) , Image.BICUBIC)
                    img_resized.save(output_image_path)
            else:
                print(f'{img_dir} not exist')






if __name__ == "__main__":
    launch_accelerate()

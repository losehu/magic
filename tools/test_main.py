import subprocess

# 定义命令和参数
command = [
    "python", 
    "tools/test.py", 
    "resume_from_checkpoint=./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400"
]

# 执行命令
result = subprocess.run(command, capture_output=True, text=True)

# 输出命令的返回结果
print(result.stdout)
if result.stderr:
    print("Error:", result.stderr)

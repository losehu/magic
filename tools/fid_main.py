import subprocess

def run_fid_score():
    # 构建命令
    command = [
        'python', 'tools/fid_score.py', 'cfg',
        'resume_from_checkpoint=/root/autodl-tmp/magic-main/magicdrive-log/SDv1.5mv-rawbox_2024-08-29_17-02_224x400/weight-E125-S72975',
        'fid.rootb=tmp/224x400/samples'
    ]
    
    # 执行命令
    try:
        subprocess.run(command, check=True)
        print("FID score calculation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_fid_score()

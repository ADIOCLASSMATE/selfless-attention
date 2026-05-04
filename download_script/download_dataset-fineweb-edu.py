import os
import time
import subprocess
import threading
import signal

def get_folder_size(path: str) -> int:
    """计算文件夹总大小（字节）"""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except FileNotFoundError:
                pass
    return total

def monitor_progress(path, stop_flag, process, no_progress_timeout=30, min_speed_kb=50):
    """
    监控文件夹下载进度。
    - 若 no_progress_timeout 秒内大小无变化，则终止进程
    - 若速率低于 min_speed_kb KB/s 持续 15 秒，则终止进程
    """
    last_size = get_folder_size(path)
    last_time = time.time()
    slow_count = 0

    while not stop_flag.is_set() and process.poll() is None:
        time.sleep(5)
        current_size = get_folder_size(path)
        elapsed = time.time() - last_time
        delta = current_size - last_size

        if delta > 0:
            speed_kb = delta / 1024 / elapsed
            print(f"📦 当前增长 {delta/1024:.1f} KB，平均速度 {speed_kb:.1f} KB/s")
            last_size = current_size
            last_time = time.time()
            slow_count = 0 if speed_kb > min_speed_kb else slow_count + 1
        else:
            print("⏸️ 暂无进度...")
            if time.time() - last_time > no_progress_timeout:
                print("⚠️ 长时间无进展，终止下载！")
                process.terminate()
                stop_flag.set()
                return

        if slow_count >= 3:  # 连续3次（15秒）速度低于阈值
            print("⚠️ 下载速度过慢，终止下载！")
            process.terminate()
            stop_flag.set()
            return


def safe_download(command, local_dir, max_retry=9090909):
    """自动重试执行 huggingface-cli download 命令"""
    retry = 0
    while retry < max_retry:
        print(f"\n🚀 开始第 {retry+1}/{max_retry} 次下载尝试...")
        stop_flag = threading.Event()
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(local_dir, stop_flag, process),
            daemon=True,
        )
        monitor_thread.start()

        # 实时打印日志
        try:
            for line in process.stdout:
                print(line.strip())
        except Exception:
            pass

        process.wait()
        stop_flag.set()
        monitor_thread.join()

        if process.returncode == 0:
            print("✅ 下载成功！")
            return
        else:
            print(f"⚠️ 下载失败或被中断（return code {process.returncode}），5秒后重试...")
            retry += 1
            time.sleep(5)

    print("❌ 多次重试仍失败，终止下载。")


if __name__ == "__main__":
    local_dir = "./dataset/open_r1_math/"
    os.makedirs(local_dir, exist_ok=True)
    access_token = "your_huggingface_access_token_here"

    command = f"huggingface-cli download open_r1_math --token {access_token} --repo-type dataset --local-dir ./dataset/open_r1_math"
    safe_download(command, local_dir)

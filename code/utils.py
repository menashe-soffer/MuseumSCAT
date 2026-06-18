import os
import sys
import subprocess


def clear_gpu_zombies():
    print("Checking for zombie CUDA processes...")
    try:
        # Run nvidia-smi to get a list of PIDs using the GPU
        pid_lines = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip().split("\n")

        current_pid = os.getpid()
        killed_any = False

        for pid_str in pid_lines:
            pid_str = pid_str.strip()
            if not pid_str:
                continue

            pid = int(pid_str)
            # Kill the process ONLY if it's NOT our current running notebook/script
            if pid != current_pid:
                print(f"💥 Found zombie process {pid} hogging VRAM. Forcefully terminating...")
                os.system(f"kill -9 {pid}")
                killed_any = True

        if not killed_any:
            print("✅ No external zombie processes found on the GPU.")

    except Exception as e:
        print(f"Skipping process cleanup (nvidia-smi might be idle or unavailable): {e}")


# 1. Clear out any trapped memory from crashed/interrupted runs
clear_gpu_zombies()

# 2. Bind cleanly to your isolated GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 3. Safe to import torch now
import torch

# 4. Wake up a pristine CUDA environment
if torch.cuda.is_available():
    _ = torch.tensor([1.0]).cuda()
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"🚀 CUDA Initialized! Fresh VRAM Available: {free_mem / 1024 ** 3:.2f} GB / {total_mem / 1024 ** 3:.2f} GB")


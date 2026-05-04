from huggingface_hub import snapshot_download
import time
while True:
    try:
        snapshot_download(
            # repo_id="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            repo_id="llamafactory/OpenR1-Math-94k",
            local_dir="./dataset/open_r1_math",
            repo_type="dataset"
        )
        print("✅ 下载成功！")
        break
    except Exception as e:
        print(f"⚠️ 下载失败，错误信息：{e}")
        print("⏳ 5秒后重试...")
        time.sleep(5)
        
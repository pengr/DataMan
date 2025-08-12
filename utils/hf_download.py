# Method 1: snapshot_download
from huggingface_hub import snapshot_download

snapshot_download(repo_id='cerebras/SlimPajama-627B',
                  local_dir='/mnt/m6pr/data/cerebras/SlimPajama-627B',
                  repo_type='dataset',
                  local_dir_use_symlinks=False,
                  resume_download=True)

# Method 2: huggingface-cli
# Set the mirror:
export HF_ENDPOINT="https://hf-mirror.com"
# Command to download the dataset:
huggingface-cli download --repo-type dataset --resume-download cerebras/SlimPajama-627B --local-dir data/cerebras/SlimPajama-627B --local-dir-use-symlinks False

# Method 3: huggingface-cli Python Code
import os
# Set environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download the dataset
os.system('huggingface-cli download --repo-type dataset --resume-download cerebras/SlimPajama-627B --local-dir data/cerebras/SlimPajama-627B --local-dir-use-symlinks False')

import os
from transformers import BertTokenizer, BertForSequenceClassification
import shutil

# Set the target directory for your local model
model_dir = os.path.join(os.path.dirname(__file__), "local-bert-model")

# Download to a temporary cache directory (Hugging Face default)
tmp_cache = os.path.join(model_dir, "tmp_hf_cache")
os.makedirs(tmp_cache, exist_ok=True)

print("Downloading BERT model and tokenizer to temporary cache...")
tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-imdb', cache_dir=tmp_cache)
model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb', cache_dir=tmp_cache)

# Find the snapshot directory
snapshots_dir = os.path.join(tmp_cache, "models--textattack--bert-base-uncased-imdb", "snapshots")
if not os.path.exists(snapshots_dir):
    raise RuntimeError("Could not find snapshot directory after download.")

snapshot_subdirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)]
snapshot_subdirs = [d for d in snapshot_subdirs if os.path.isdir(d)]
if not snapshot_subdirs:
    raise RuntimeError("No snapshot subdirectory found after download.")

snapshot_dir = snapshot_subdirs[0]  # There should only be one

# Copy all files from the snapshot to the target model_dir
os.makedirs(model_dir, exist_ok=True)
for fname in os.listdir(snapshot_dir):
    src = os.path.join(snapshot_dir, fname)
    dst = os.path.join(model_dir, fname)
    shutil.copy2(src, dst)

print(f"Copied model files to {model_dir}")

# Optionally, clean up the temporary cache
shutil.rmtree(tmp_cache)
print("Temporary cache cleaned up. Local BERT model is ready.") 
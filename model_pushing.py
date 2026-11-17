# model_pushing.py
# pip install -U transformers huggingface_hub

import os
from huggingface_hub import login, HfApi, create_repo, upload_folder, upload_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- EDIT THESE TWO ---
LOCAL_DIR = "fine_tuned_ad_classification_xlm"
REPO_ID   = "impresso-project/impresso-ad-classification-xlm-one-class"
# ----------------------

# Optional extra files to include alongside the weights
EXTRA_FILES = [
    "pipeline.py",               # your custom pipeline
    "best_params_final.json",    # defaults used by the pipeline (optional)
    "README.md",                 # model card (optional but recommended)
    "requirements.txt",          # if you have runtime deps (optional)
    "meta_classifier.pkl",       # if you use the meta-stacker (optional)
    "LICENSE",                   # if you have one (optional)
]

def main():
    # 1) Auth
    login()  # paste your HF token when prompted

    # 2) Create (or reuse) the Hub repo
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # 3) Quick sanity check that the dir is a valid Transformers model
    _ = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    _ = AutoTokenizer.from_pretrained(LOCAL_DIR)

    # 4) Upload model + tokenizer files from LOCAL_DIR
    #    (keeps only what a user needs; skips big training checkpoints)
    upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=LOCAL_DIR,
        commit_message="Upload fine-tuned model and tokenizer",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
            "training_args.bin",
            "training_info.json",
        ],
        ignore_patterns=[
            "checkpoint-*",
            "runs/*",
            "events.out.tfevents*",
            "*.pt.tmp",
            "*.bin.tmp",
        ],
    )

    # 5) Upload extra files (pipeline.py, README, etc.) to the repo root
    for path in EXTRA_FILES:
        if os.path.exists(path):
            upload_file(
                repo_id=REPO_ID,
                repo_type="model",
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                commit_message=f"Add {os.path.basename(path)}",
            )

    print(f"Done! Repo: {REPO_ID}\n"
          f"Load with:\n"
          f"  from transformers import pipeline\n"
          f"  pipe = pipeline('text-classification', model='{REPO_ID}', trust_remote_code=True)")

if __name__ == "__main__":
    main()

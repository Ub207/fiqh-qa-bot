from huggingface_hub import HfApi, create_repo

REPO_ID = "ubaid-ai/fiqh-qa-bot-data"
REPO_TYPE = "dataset"

create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
api = HfApi()

files = [
    ("faiss_index/fiqh.index", "fiqh.index"),
    ("faiss_index/chunks.pkl", "chunks.pkl"),
    ("fiqh_data/fiqh_qa.json", "fiqh_qa.json"),
    ("fiqh_data/fiqh_categories.json", "fiqh_categories.json"),
]

for local, remote in files:
    print(f"Uploading {local} -> {remote}")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"  Done!")

print(f"\nAll uploaded! https://huggingface.co/datasets/{REPO_ID}")

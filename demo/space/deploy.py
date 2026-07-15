"""Deploy the ProtoSleepNet demo: private data repo + public Docker Space proxy.

Run AFTER `huggingface-cli login` (write token). Creates/updates:
  - a PRIVATE dataset repo holding the derived bundle
  - a PUBLIC Docker Space serving the built app + an authenticated /data proxy
and sets the Space's DATASET_REPO variable. The HF_TOKEN read secret is added
only if --read-token is given; otherwise add it in the Space UI (Settings →
Secrets) so the value never passes through the shell history.

Usage:
  python deploy.py --dataset-repo 4rooms/protosleepnet-demo-data \
                   --space-repo   4rooms/protosleepnet-demo \
                   --bundle /path/to/bundle_indomain --space-dir .
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--space-repo", required=True)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--space-dir", required=True, type=Path)
    ap.add_argument("--read-token", default=None, help="optional read token to set as the HF_TOKEN secret")
    args = ap.parse_args()
    api = HfApi()

    # 1) private dataset repo + bundle
    api.create_repo(args.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"[data] uploading bundle → {args.dataset_repo} (private) …")
    api.upload_folder(folder_path=str(args.bundle), repo_id=args.dataset_repo,
                      repo_type="dataset", commit_message="demo bundle (derived, in-domain)")

    # 2) public Docker Space (app + proxy)
    api.create_repo(args.space_repo, repo_type="space", space_sdk="docker",
                    private=False, exist_ok=True)
    print(f"[space] uploading app + proxy → {args.space_repo} (public) …")
    api.upload_folder(folder_path=str(args.space_dir), repo_id=args.space_repo,
                      repo_type="space", ignore_patterns=["deploy.py", ".gitignore", "**/.DS_Store"],
                      commit_message="docker space: static app + authenticated data proxy")

    # 3) wire the Space to the private data repo
    api.add_space_variable(args.space_repo, "DATASET_REPO", args.dataset_repo)
    if args.read_token:
        api.add_space_secret(args.space_repo, "HF_TOKEN", args.read_token)
        print("[space] HF_TOKEN secret set")
    else:
        print("[space] NOTE: add the HF_TOKEN read secret in Space Settings → Secrets")

    print(f"\n✅ Space:   https://huggingface.co/spaces/{args.space_repo}")
    print(f"   Dataset: https://huggingface.co/datasets/{args.dataset_repo} (private)")


if __name__ == "__main__":
    main()

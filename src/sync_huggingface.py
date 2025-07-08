# this script syncronize the models/ directory with the Hugging Face Hub  repo 4rooms/ProtoEx

# upload routine

import os 
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

def upload_models():
    """Upload models to Hugging Face Hub"""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Please set it with: export HF_TOKEN=your_token_here")
        return False
    
    print("Using Hugging Face token:", token)

    try:
        api = HfApi(token=token)
        print("Uploading models to Hugging Face Hub...")
        api.upload_large_folder(
            folder_path="models/",
            repo_id="4rooms/ProtoEx",
            repo_type="model",
        )
        print("Upload completed successfully!")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please check your HF_TOKEN.")
            print("Make sure your token has write permissions for the repository.")
        else:
            print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def download_models():
    """Download models from Hugging Face Hub"""
    try:
        print("Downloading models from Hugging Face Hub...")
        snapshot_download(
            repo_id="4rooms/ProtoEx",
            repo_type="model",
            local_dir="models/",
            local_dir_use_symlinks=False
        )
        print("Download completed successfully!")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print("Error: Repository not found. Please check the repository ID.")
        else:
            print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Sync models with Hugging Face Hub")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload models to Hugging Face Hub",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download models from Hugging Face Hub",
    )

    args = parser.parse_args()

    if args.upload:
        success = upload_models()
        exit(0 if success else 1)
    elif args.download:
        success = download_models()
        exit(0 if success else 1)
    else:
        print("Please specify --upload or --download")
        parser.print_help()
        exit(1)



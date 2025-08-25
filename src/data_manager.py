import tarfile
from pathlib import Path

import requests


class DataManager:
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    def download_and_extract_mmlu(self) -> bool:
        if self.data_dir.exists():
            print(f"Data directory {self.data_dir} already exists, skipping download")
            return True

        print(f"Downloading MMLU data from {self.data_url}")
        tar_path = self.base_dir / "data.tar"

        try:
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"Progress: {progress:.1f}%", end="\r")

            print("\nExtracting data...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(self.base_dir)

            print(f"Data extracted to {self.data_dir}")
            return True

        except Exception as e:
            print(f"Error downloading/extracting data: {e}")
            return False

        finally:
            if tar_path.exists():
                tar_path.unlink()

    def prepare_llm_data_dir(self, llm_name: str) -> Path:
        llm_data_dir = self.base_dir / f"data_{llm_name.replace(':', '')}"
        llm_data_dir.mkdir(exist_ok=True)

        test_src = self.data_dir / "test"
        test_dst = llm_data_dir / "test"

        if not test_dst.exists() and test_src.exists():
            import shutil

            shutil.copytree(test_src, test_dst)
            print(f"Copied test data to {test_dst}")

        return llm_data_dir


import os

from medmnist import INFO
import medmnist


def download_bloodmnist(data_dir: str = "./data") -> None:
    """
    Download the BloodMNIST dataset (train/val/test) to the given directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    info = INFO["bloodmnist"]
    DataClass = getattr(medmnist, info["python_class"])

    # This will download the data for each split into data_dir/bloodmnist
    for split in ["train", "val", "test"]:
        DataClass(root=data_dir, split=split, transform=None, download=True)

    print(f"BloodMNIST downloaded to: {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    download_bloodmnist()


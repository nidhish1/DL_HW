import os

from torchvision import datasets, transforms


def download_mnist(data_dir: str = "./data") -> None:

    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.ToTensor()

    datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    print(f"MNIST downloaded to: {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    download_mnist()


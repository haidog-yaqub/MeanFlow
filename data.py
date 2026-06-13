import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_dataset(cfg):
    if cfg["dataset"] == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=cfg["data_root"],
            train=True,
            download=True,
            transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
        )

    if cfg["dataset"] == "mnist":
        return torchvision.datasets.MNIST(
            root=cfg["data_root"],
            train=True,
            download=True,
            transform=T.Compose([
                T.Resize((cfg["image_size"], cfg["image_size"])),
                T.ToTensor(),
            ]),
        )

    if cfg["dataset"] == "imagenet":
        transform = T.Compose([
            T.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg["image_size"])),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        return ImageFolder(cfg["data_root"], transform=transform)

    raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item

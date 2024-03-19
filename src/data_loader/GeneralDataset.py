import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GeneralDataset(Dataset):
    """Dataset for image items

    Args:
        file_names: list of paths to images
        transform: boolean indicator if images should be centered channel wise with mean and standard deviation, must be given when True, default: False
        img_size: size of images to be loaded, images will be transformed to given size
        mean_train_set: array of shape (3,) with channel wise mean of train set
        std_train_set: array of shape (3,) with channel wise standard deviation of train set
        sw: bool, whether the dataset contains black and white or rgb images
    """

    def __init__(
        self,
        file_names: list[str],
        transform: bool = False,
        img_size: int = 512,
        mean: np.ndarray = np.array([0, 0, 0]),
        std: np.ndarray = np.array([1, 1, 1]),
        validation: bool = False,
        only_labels: bool = False,
    ) -> None:
        self.file_names = file_names
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.validation = validation
        self.only_labels = only_labels
        self.image_transform = (
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ],
            )
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ],
            )
        )
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ],
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        """Function to get one image of dataset at a given index

        Args:
            index: index of image to be loaded

        Return: tensor of loaded image
        """
        image_path = self.file_names[index]

        if self.only_labels:
            if os.path.dirname(image_path).endswith("good") or os.path.dirname(
                image_path
            ).endswith("ok"):
                target = 0
            else:
                target = 1

            return target

        image = Image.open(image_path).convert("RGB")

        image = self.image_transform(image)

        if not self.validation:
            return image

        if os.path.dirname(image_path).endswith("good") or os.path.dirname(
            image_path
        ).endswith("ok"):
            pixel_target = torch.zeros([1, image.shape[1], image.shape[2]])
            target = 0
        elif os.path.dirname(image_path).endswith("ko"):
            if "/03/" in image_path:
                pixel_target = Image.open(image_path.replace("/test/", "/ground_truth/"))
            else:
                pixel_target = Image.open(
                    image_path.replace("/test/", "/ground_truth/").replace(".bmp", ".png")
                )
            pixel_target = self.target_transform(pixel_target)
            # make sure only the value 0 and 255 exist
            pixel_target[pixel_target != 0] = 1
            target = 1
        else:
            pixel_target = Image.open(
                image_path.replace("/test/", "/ground_truth/")
                .replace(".png", "_mask.png")
                .replace(".bmp", ".png")
            )
            pixel_target = self.target_transform(pixel_target)
            # make sure only the value 0 and 255 exist
            pixel_target[pixel_target != 0] = 1
            target = 1
        return image, pixel_target, target

    def __len__(self) -> int:
        return len(self.file_names)

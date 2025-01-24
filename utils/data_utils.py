import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import torchvision
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from pathlib import Path
import json
import math


class CustomDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(
            self, 
            img_size: Tuple, 
            root: str, 
            split: str, 
            download: bool, 
            transform: torchvision.transforms, 
            target_types: List
            ):
        super().__init__(
            root=root, 
            split=split, 
            download=download, 
            transform=transform, 
            target_types=target_types
            )
        self.img_size = img_size

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        image = Image.open(self._images[idx]).convert("RGB")
        # 1 = background, 2 = pet, 3 = border
        seg_mask = Image.open(self._segs[idx]).resize(self.img_size)
        array_seg_mask = np.array(seg_mask)
        array_seg_mask[array_seg_mask == 1] = 0
        array_seg_mask[array_seg_mask == 2] = 1
        array_seg_mask[array_seg_mask == 3] = 1

        if self.transforms:
            image, seg_mask = oxford_transform(
                image=image, 
                seg_mask=array_seg_mask, 
                img_size=self.img_size
                )

        return image, seg_mask


def oxford_transform(
        image: Image, 
        seg_mask: np.array, 
        img_size: Tuple
        ) -> Tuple[torch.tensor, torch.tensor]:
    '''
    transform image and segmentation mask into tensors and resize
    '''
    image = TF.to_tensor(pic=image)
    seg_mask = torch.tensor(seg_mask)
    image = TF.resize(image, size=img_size)

    return image, seg_mask


def create_dataloaders(
        batch_size: int, 
        img_size: Tuple, 
        data_root: str
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    creates and partitions train, validation and test data loaders
    '''
    trainval_dataset = CustomDataset(
        root=data_root,
        split="trainval",
        download=True, # will skip downloading if already downloaded
        transform=True,
        target_types=["segmentation", "category"],
        img_size=img_size
    )
    test_dataset = CustomDataset(
        root=data_root,
        split="test",
        download=True, # will skip downloading if already downloaded
        transform=True,
        target_types=["segmentation", "category"],
        img_size=img_size
    )
    # combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([trainval_dataset, test_dataset])

    # assign dataset sizes
    train_amount = int(len(combined_dataset)*0.9)
    val_amount = int((len(combined_dataset) - train_amount) / 2)
    test_amount = len(combined_dataset) - train_amount - val_amount

    # randomly split datasets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, 
        lengths=[train_amount, val_amount, test_amount]
        )
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_train_results(
        train_loss_list: List, 
        val_loss_list: List,
        file_path: Path,
        continue_bool: bool,
        batch_size: int,
        learning_rate: float
        ) -> None:
    '''
    saves results from developing model into a JSON file
    '''
    if continue_bool:
        with open(file_path, mode="r") as opened_json:
            json_dict = json.load(opened_json)
        with open(file_path, mode="w") as opened_json:
            if not "batch size" in list(json_dict.keys()):
                json_dict["batch size"] = batch_size
            if not "learning rate" in list(json_dict.keys()):
                json_dict["learning rate"] = learning_rate
            
            json_dict["train loss"] += train_loss_list
            json_dict["validation loss"] += val_loss_list
            json_obj = json.dumps(json_dict)
            opened_json.write(json_obj)
    else:
        with open(file_path, mode="w") as opened_json:
            json_dict = {
                "train loss": train_loss_list,
                "validation loss": val_loss_list,
                "batch size": batch_size,
                "learning rate": learning_rate
            }
            json_dict = json.dumps(json_dict)
            opened_json.write(json_dict) 


def save_test_results(
        test_dict: Dict, 
        file_path: Path
        ) -> None:
    '''
    saves results from developing model into a JSON file
    '''
    with open(file_path, mode="w") as opened_json:
        json_obj = json.dumps(test_dict)
        opened_json.write(json_obj) 


def calc_iou(batch_pred, batch_mask, mask):
    intersection = (batch_pred.bool() & batch_mask.bool()).float().sum().item()
    union = (batch_pred.bool() | batch_mask.bool()).float().sum().item()

    pred_masked = batch_pred.bool() & mask
    target_masked = batch_mask.bool() & mask
    intersection_masked = (pred_masked & target_masked).float().sum().item()
    union_masked = (pred_masked | target_masked).float().sum().item()

    return intersection, intersection_masked, union, union_masked


def calc_correct(batch_pred, batch_mask, mask):
    correct = (batch_pred == batch_mask)
    correct_masked = correct & mask

    total_correct = correct.sum().item()
    total_pixels = torch.numel(batch_pred)

    total_correct_masked = correct_masked.sum().item()
    total_pixels_masked = mask.sum().item()

    return total_correct, total_pixels, total_correct_masked, total_pixels_masked

import torchvision
from pathlib import Path
import torch
from typing import Dict


def define_model(
        num_classes: int
        ) -> torchvision.models.segmentation.fcn_resnet101:
    '''
    defines the model architecture
    '''
    model = torchvision.models.segmentation.fcn_resnet101(
        num_classes=21, 
        )
    
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
	
    return model 


def get_num_params(tensor_size: torch.Tensor.size) -> int:
    num_params = 1
    for size_idx in tensor_size:
        num_params *= size_idx
    return num_params


def print_model_summary(
        model: torchvision.models.segmentation.fcn_resnet101
        ) -> None:
    '''
    prints the parameters and parameter size
    should contain an equal number of trainable and non-trainable
    parameters since the teacher network parameters are not updated
    with gradient descent
    '''
    trainable = 0
    non_trainable = 0
    for param in model.named_parameters():
        if param[1].requires_grad:
            trainable += get_num_params(param[1].size())
        else:
            non_trainable += get_num_params(param[1].size())
        print(param[0], param[1].size(), param[1].requires_grad)

    print("trainable parameters:", trainable)
    print("non-trainable parameters:", non_trainable)


def load_model(
        weight_path: Path, 
        model: torchvision.models.segmentation.fcn_resnet101
        ) -> torchvision.models.segmentation.fcn_resnet101:
    '''
    loads all parameters of a model
    '''
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint)
    return model


def define_optimizer(
        model: torchvision.models.segmentation.fcn_resnet101, 
        learning_rate: float
        ) -> torch.optim.Adam:    
    '''
    returns optimizer
    '''     
    return torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def define_criterion() -> torch.nn.BCEWithLogitsLoss:
    '''
    returns loss function
    '''
    return torch.nn.BCEWithLogitsLoss()


def define_device() -> torch.device:
    '''
    returns torch device
    '''
    return torch.device("cuda")


def save_checkpoint(state: Dict, filepath: Path) -> None:
    '''
    saves the model state dictionary to a .pth.tar tile
    '''
    print("saving...")
    torch.save(state, filepath)

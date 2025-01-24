import argparse
import torchvision
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm
from utils.data_utils import *
from utils.model_utils import *


def create_argparser() -> argparse.Namespace:
    '''
    defines the command line argument parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained", action="store_true", default=True) 
    parser.add_argument("-num_classes", type=int, default=1)
    parser.add_argument("-batch_size", default=32)
    parser.add_argument("-img_size", default=(128, 128))
    parser.add_argument("-patience", default=3)
    parser.add_argument("-result_dir", type=Path, default=Path("C:\\personal_ML\\basic-semantic-segmentation\\results\\"))
    parser.add_argument("-train_result_filename", type=str, default="model_3_train_results.json")
    parser.add_argument("-test_result_filename", type=str, default="model_3_test_results.json")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-model_save_name", default="model_3.pth.tar")
    parser.add_argument("-num_epochs", default=64)
    parser.add_argument("-data_root", default=Path("C:\\personal_ML\\Oxford_PyTorch\\"))
    parser.add_argument("-continue_bool", action="store_true", default=False)
    parser.add_argument("-start_epoch", type=int, default=0)
    parser.add_argument("-weight_path", default=Path("C:\\personal_ML\\basic-semantic-segmentation\\results\\model_2.pth.tar"))
    return parser.parse_args()


def evaluate(
        loader: DataLoader, 
        model: torchvision.models.segmentation.fcn_resnet101, 
        device: torch.device, 
        criterion: torch.optim, 
        ) -> torch.tensor:
    '''
    validates model
    '''
    with torch.no_grad():
        model.eval() 
        loss = 0
        
        total_correct = 0
        total_pixels = 0
        total_correct_masked = 0
        total_pixels_masked = 0
        
        intersection = 0
        union = 0
        intersection_masked = 0
        union_masked = 0
        
        for batch_image, batch_mask in tqdm(loader, desc="Evaluating"):
            batch_pred = model(batch_image.to(device))
            batch_pred = batch_pred["out"].squeeze(1)
            batch_mask = batch_mask.float().to(device)
            loss += criterion(batch_pred, batch_mask).item()
            
            batch_pred = (torch.sigmoid(batch_pred) > 0.5).float()

            mask = batch_mask != 0 
            correct = (batch_pred == batch_mask)
            correct_masked = correct & mask
            
            total_correct += correct.sum().item()
            total_pixels += torch.numel(batch_pred)

            total_correct_masked += correct_masked.sum().item()
            total_pixels_masked += mask.sum().item()

            intersection += (batch_pred.bool() & batch_mask.bool()).float().sum().item()
            union += (batch_pred.bool() | batch_mask.bool()).float().sum().item()

            inter, inter_masked, un, un_masked = calc_iou(batch_pred=batch_pred, batch_mask=batch_mask, mask=mask)
            tot_corr, num_pixels, tot_correct_masked, num_pixels_masked = calc_correct(batch_pred=batch_pred, batch_mask=batch_mask, mask=mask)

            intersection += inter
            union += un
            intersection_masked += inter_masked
            union_masked += un_masked

            total_correct += tot_corr
            total_pixels += num_pixels
            total_correct_masked += tot_correct_masked
            total_pixels_masked += num_pixels_masked            

        pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
        pixel_accuracy_masked = total_correct_masked / total_pixels_masked if total_pixels_masked > 0 else 0

        iou = intersection / union if union > 0 else 0
        iou_masked = intersection_masked / union_masked if union_masked > 0 else 0
        loss /= len(loader)
    
    return loss, pixel_accuracy, pixel_accuracy_masked, iou, iou_masked


def train(
        model: torchvision.models.segmentation.fcn_resnet101, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int, 
        patience: int, 
        criterion: torch.nn.BCEWithLogitsLoss, 
        optimizer: torch.optim, 
        device: torch.device, 
        result_dir: Path, 
        model_save_name: str,
        continue_bool: bool,
        start_epoch: int,
        ) -> Tuple[List, List]:
    ''' 
    trains model and records training and validation loss throughout training
    '''

    patience_counter = 0
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    if continue_bool:
        num_epochs += start_epoch
        print("epoch range", start_epoch, num_epochs)

    for epoch_idx in range(start_epoch, num_epochs):
        if patience == patience_counter:
            break
        else:
            epoch_loss = 0
            model.train()

            for batch_image, batch_seg_mask in tqdm(train_loader, desc="Training"):
                    batch_pred = model(batch_image.to(device))["out"]
                    loss = criterion(batch_pred.squeeze(1), batch_seg_mask.float().to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            train_loss_list.append((epoch_idx+1, epoch_loss/len(train_loader)))
            val_loss, pa, pa_masked, iou, iou_masked = evaluate(val_loader, model, criterion=criterion, device=device)
            print(f"Epoch {epoch_idx+1} Validation Loss", val_loss, "Train Loss:", epoch_loss/len(train_loader))
            print(f"Pixel Accuracy: {pa}, Pixel Accuracy (masked): {pa_masked}")
            print(f"IOU: {iou}, IOU (masked): {iou_masked}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(state=model.state_dict(), filepath=result_dir.joinpath(model_save_name))
            else:
                patience_counter += 1  
    
    return train_loss_list, val_loss_list   


def test_model(
        test_loader: DataLoader, 
        model: torchvision.models.segmentation.fcn_resnet101, 
        device: torch.device, 
        criterion: torch.nn.BCEWithLogitsLoss,
        result_dir: Path, 
        model_save_name: str
        ) -> Dict:
    '''
    measures model performance on the test dataset
    '''
    with torch.no_grad():
        model = load_model(weight_path=result_dir.joinpath(model_save_name), model=model)
        _, pixel_accuracy, pixel_accuracy_masked, iou, iou_masked = evaluate(loader=test_loader, model=model, device=device, criterion=criterion)
    return pixel_accuracy, pixel_accuracy_masked, iou, iou_masked
    

def main():
    '''
    trains, validates and tests resnet FCN
    '''
    args = create_argparser()
    model = define_model(num_classes=args.num_classes)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size, img_size=args.img_size, data_root=args.data_root
        )
    criterion = define_criterion()
    optimizer = define_optimizer(model=model, learning_rate=args.lr)
    device = define_device()
    model = model.to(device)

    if args.continue_bool:
        model = load_model(weight_path=args.weight_path, model=model)
    print_model_summary(model=model)
    train_loss_list, val_loss_list = train(
        model=model, 
        train_loader=train_loader,  
        val_loader=val_loader, 
        num_epochs=args.num_epochs,
        patience=args.patience,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        result_dir=args.result_dir,
        model_save_name=args.model_save_name,
        continue_bool=args.continue_bool,
        start_epoch=args.start_epoch,
        )
    save_train_results(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        file_path=args.result_dir.joinpath(args.train_result_filename),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        continue_bool=args.continue_bool
    )
    pixel_accuracy, pixel_accuracy_masked, iou, iou_masked = test_model(
        test_loader=test_loader, 
        model=model, 
        device=device, 
        result_dir=args.result_dir,
        criterion=criterion,
        model_save_name=args.model_save_name
        )
    test_dict = {
        "pixel accuracy": pixel_accuracy,
        "pixel accuracy (masked)": pixel_accuracy_masked,
        "iou": iou,
        "iou (masked)": iou_masked,
    }
    print(f"Test Results: {test_dict}")
    save_test_results(
        file_path=args.result_dir.joinpath(args.test_result_filename),
        test_dict=test_dict
    )


if __name__ == "__main__":
    main()

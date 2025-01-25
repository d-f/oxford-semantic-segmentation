Using a fully convolutional network with a ResNet101 backbone, a model was trained to achieve the following performance metrics on the Oxford-IIIT Pet dataset.

| IoU  | IoU (excluding background class) | Pixel Accuracy | Pixel Accuracy (excluding background class) |
| ---- | -------------------------------- | -------------- | ------------------------------------------- |
| 0.93 | 0.96                             | 0.95           | 0.96                                        |

The model weights were initialized using weights previously trained on the 20 COCO categories that are present in the Pascal VOC dataset. [https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.FCN_ResNet101_Weights](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.FCN_ResNet101_Weights)

All model parameters were allowed to update during training.

The images were resized to (128, 128) and the pixel values were normalized between 0 and 1. Better results may have been achieved if the values were z-score normalized.

The dataset was divided into partitions of 5% for validation, 5% for testing and 90% for training.

| Training images | Validation images | Test images | 
| --------------- | ----------------- | ----------- |
| 6614            | 367               | 368         |

The model was trained using a batch size of 32, a learning rate of 1 × 10<sup>-3</sup> for 16 epochs where early stopping with a patience criteria of 3 epochs was used to stop the training when the validation loss did not decrease.

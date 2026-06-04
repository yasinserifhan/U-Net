
# PyTorch U-Net Image Segmentation

This repository contains a PyTorch implementation of the U-Net architecture for image segmentation. The project is configured out-of-the-box for a Carvana-style dataset but can be adapted for other binary segmentation tasks.

## Features

* **Custom U-Net Architecture:** Built from scratch using PyTorch, featuring `DoubleConv` blocks, downsampling, and upsampling with skip connections.
* **Robust Data Pipeline:** Custom `CarvanaDataset` class that automatically converts raw images to RGB, processes mask images to grayscale, and binarizes the mask values.
* **Data Augmentation:** Utilizes the `albumentations` library for dynamic data augmentation, including resizing, rotations, horizontal/vertical flips, and normalization.
* **Mixed Precision Training:** Implements `torch.cuda.amp.GradScaler` to significantly speed up the training process and reduce memory usage.
* **Automatic Data Splitting:** Uses `scikit-learn` to automatically split the dataset into a 90% training and 10% validation set.
* **Comprehensive Metrics:** Calculates Pixel Accuracy and Dice Score during the validation phase to strictly monitor model performance.
* **Checkpoint Management:** Includes utility functions to save and load model states to easily resume training.

## Project Structure

* `model.py`: Contains the definition of the U-Net architecture (`UNet` and `DoubleConv` classes) and a simple testing script.
* `dataset.py`: Defines the custom PyTorch `Dataset` used to load images and their corresponding mask files.
* `train.py`: The main execution script that defines hyperparameters, loss functions (`BCEWithLogitsLoss`), the optimizer (`Adam`), and runs the training/validation loops.
* `utils.py`: Helper functions for generating data loaders, computing evaluation metrics, and saving prediction samples as images.

## Configuration & Hyperparameters

The training process is controlled by variables defined in `train.py`:
* **Learning Rate:** `1e-4`
* **Batch Size:** `16`
* **Epochs:** `3`
* **Image Dimensions:** Resized to `320x480` (original typically 1280x1918)
* **Device:** Automatically detects and uses `cuda` if available, otherwise defaults to `cpu`.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yasinserifhan/U-Net.git](https://github.com/yasinserifhan/U-Net.git)
   cd U-Net

```

2. **Install dependencies:**
Ensure you have PyTorch installed. You will also need:
```bash
pip install albumentations scikit-learn tqdm Pillow numpy torchvision

```


3. **Prepare the Data:**
By default, the script looks for your dataset in the following directories:
* Images: `data/train/`
* Masks: `data/train_masks/`


*Note: Mask images are expected to have a `_mask.gif` suffix corresponding to their `.jpg` image counterparts.*

## Usage

To start training the model, simply run the training script:

```bash
python train.py

```

During training, the script will:

1. Display a progress bar with the current loss using `tqdm`.
2. Evaluate the model on the validation set after each epoch, printing the Pixel Accuracy and Dice Score.
3. Save the model checkpoint to `my_checkpoint.pth.tar`.
4. Save sample prediction overlays into a `saved_images/` folder.

```

```

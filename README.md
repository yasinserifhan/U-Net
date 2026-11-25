# U-Net from scratch in PyTorch

A from-scratch PyTorch implementation of U-Net for binary image segmentation, set up for the Carvana Image Masking Challenge data (car vs. background masks). Swap the dataset class to use it for other binary segmentation tasks.

<!-- Bir tutorial'dan yola çıktıysan buraya ekle: "Based on <tutorial adı + link>; my changes: 320×480 input, train/val split with scikit-learn, ..." -->

## What's in the code

| File | Contents |
|---|---|
| `model.py` | U-Net: four encoder stages of `DoubleConv` (two 3×3 conv → BatchNorm → ReLU) with 64 / 128 / 256 / 512 channels, a 1024-channel bottleneck, a transposed-conv decoder with skip connections, and a 1×1 output conv. If a decoder feature map and its skip connection differ in size, the decoder output is resized, so odd input sizes work. `python model.py` runs a shape test on a 161×161 input. |
| `dataset.py` | `CarvanaDataset`: loads RGB images and grayscale masks, binarizes masks to {0, 1}. |
| `train.py` | `BCEWithLogitsLoss`, Adam (lr 1e-4), mixed-precision training with a gradient scaler, albumentations augmentation (resize to 320×480, rotation up to ±35°, horizontal and vertical flips). |
| `utils.py` | 90/10 train/validation split (scikit-learn, fixed seed), data loaders, checkpoint save/load, pixel accuracy and Dice on the validation set, predicted masks saved as PNG. |

## Data

Not included. Download the training images and masks from the Kaggle **Carvana Image Masking Challenge** and arrange them as:

```text
data/
├── train/         # car_id.jpg
└── train_masks/   # car_id_mask.gif
```

## Run

```bash
pip install torch torchvision albumentations scikit-learn tqdm Pillow numpy
python train.py
```

Defaults (top of `train.py`): batch size 16, 3 epochs, 320×480 input, CUDA when available. After each epoch the script prints validation pixel accuracy and Dice, saves `my_checkpoint.pth.tar`, and writes predicted masks to `saved_images/`.

## Results

<!-- Bir eğitim çalıştırınca doldur: epoch sayısı, doğrulama Dice ve pixel accuracy, bir örnek görsel (görüntü | gerçek maske | tahmin).
     Çalıştırmadan sayı yazma; bu bölümü o zamana kadar silebilirsin. -->

## Limitations

- Evaluation uses only the 10% validation split; there is no separate test set or early stopping.
- Training is short by default (3 epochs).

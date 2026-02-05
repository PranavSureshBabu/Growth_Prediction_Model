# Grass Monitoring System

A CNN-based classifier that distinguishes between **trimmed** and **overgrown** lawns using a fine-tuned MobileNetV2 model with color-based grass detection preprocessing.

## Features

- Grass region detection using Excess Green Index + HSV filtering
- Transfer learning with MobileNetV2 (frozen backbone, trainable classifier)
- Automatic cropping to grass regions for better classification accuracy

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster training)

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

Before training, organize your dataset in the following structure:

```
dataset/
├── trimmed/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── grown/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

- **trimmed/**: Images of well-maintained, cut lawns
- **grown/**: Images of overgrown, uncut lawns
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Usage

### Training the Model

1. Place your training images in the `dataset/` folder following the structure above
2. Run the training script:

```bash
python train_model.py
```

The model will be saved as `grass_cnn.pth` after training completes.

**Training Configuration** (edit `train_model.py` to adjust):
- `VAL_FRACTION`: Validation split ratio (default: 0.2)
- `BATCH_SIZE`: Training batch size (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 15)
- `LR`: Learning rate (default: 1e-3)

### Making Predictions

To classify a new image:

```bash
python predict_model.py path/to/your/image.jpg
```

**Example output:**
```
path/to/your/image.jpg → grown (P(trimmed)=0.15, P(grown)=0.85)
```

### Where to Put Unseen Data

For prediction on new/unseen images:
- Place images **anywhere** on your system
- Pass the full or relative path to the prediction script
- No specific folder structure required for inference

**Example:**
```bash
# Using relative path
python predict_model.py data/test_lawn.jpg

# Using absolute path
python predict_model.py C:/Users/photos/lawn_photo.jpg
```

## Model Output

- `grass_cnn.pth`: Trained model weights (generated after training)

## Notes

- The model automatically detects and crops to the grass region in images
- GPU acceleration is used automatically if CUDA is available
- For best results, use clear images with visible grass areas

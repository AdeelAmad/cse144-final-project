# CSE144 Final Project - Image Classification Pipeline

## Description

This repo implements a complete image classification pipeline using a pre-trained EfficientNet_V2_L model in PyTorch. It automatically downloads the required dataset from Kaggle (`ucsc-cse-144-winter-2026-final-project`), trains the model using data augmentation techniques like MixUp and CutMix, and generates predictions for submission to the competition.

## File Structure

* **`main.py`**: The main entry point to initiate the training process.
* **`manual_test.py`**: The entry point to run inference on a saved model checkpoint.
* **`model.py`**: Defines the EfficientNet model architecture and setups up various training stages.
* **`trainer.py`**: Contains the training loop, validation logic, checkpoint saving, and curve plotting.
* **`tester.py`**: Handles inference on the unlabeled test dataset and generates output predictions.
* **`dataloader.py`**: Authenticates with Kaggle, downloads the dataset, and sets up data loaders and transformations.
* **`visualizer.py`**: Generates visual representations and summaries of the model's architecture.
* **`submission.csv`**: The final output file containing generated labels for the test images.

## Requirements

* Python 3.x
* PyTorch & Torchvision
* Kagglehub (for dataset downloading)
* Matplotlib (for training curves)
* Pandas (for CSV generation)
* Pillow (for image processing)
* Tqdm (for progress bars)
* Torchviz & Torchinfo (for `visualizer.py`)

## Usage Instructions

### 1. Training the Model

Run the main script to begin the pipeline. This will automatically fetch the data, initialize the model, train for the specified number of epochs, save the best checkpoint, and plot training/validation curves. It will also run the inference and generate a submission.csv.

```bash
python main.py
```

### 2. Running Inference

If you want to stop early you can end main.py and just run the manual test to do inference on the most recently saved checkpoint.

```bash
python manual_test.py
```

### 3. Visualizing the Architecture

To print a summary of the model parameters:

```bash
python visualizer.py
```

## Configuration Notes

* **Device**: The code currently defaults to `"mps"` (Apple Metal Performance Shaders) in `main.py`, `manual_test.py`, and `visualizer.py`. Update this to `"cuda"` or `"cpu"` depending on your hardware.
* **Hyperparameters**: Batch size is set to 32, epochs to 30, and the train/validation split to 80/20 inside `main.py`. Modification of learning rate (1e-3 default) can be done in `trainer.py`.
* **Training**: The model is too big and the images are also too big to train on a standard GPU. You must use a gpu in the cloud in order to train in a reasonable manner.
# Cats vs. Dogs: A Comparative Study in Image Classification

**Team:** Zady Verdecia & Ricardo Arenas
**Course:** CAI2840C — Introduction to Computer Vision, Miami Dade College
**Instructor:** Professor Desiree Dominguez

## Description

This project compares three different computer vision approaches to classifying images of cats and dogs on the same 25,000-image Kaggle/Microsoft dataset: a classical pipeline that extracts Histogram of Oriented Gradients (HOG) features and classifies them with an RBF-kernel SVM, a convolutional neural network built and trained from scratch, and a transfer learning model based on MobileNetV2 fine-tuned from ImageNet weights. All three methods are trained and evaluated on identical train/validation/test splits (70/15/15) so their accuracy, precision, recall, F1-score, training time, and parameter count can be fairly compared, with the goal of understanding how classical feature engineering, custom deep learning, and transfer learning trade off against each other in both performance and computational cost.

## Running the Code

The notebook is built for Google Colab and must be run top to bottom, in order, in a single session:

1. **Mount Drive & set paths** — mounts Google Drive and defines the project's data/results/model directories there, so progress survives a Colab disconnect.
2. **Setup & data cleaning** — downloads and extracts the dataset, scans every image with PIL, and removes any corrupted files.
3. **Dataset split** — defines and runs the function that splits the data into train/val/test folders.
4. **HOG feature extraction** — extracts HOG features in parallel across CPU cores, with checkpointing so a disconnect only costs the split that was in progress.
5. **HOG + SVM** — hyperparameter search, model fitting, and evaluation.
6. **Custom CNN** — model definition, training with early stopping, and evaluation.
7. **MobileNetV2 transfer learning** — two-phase training (frozen base, then fine-tuning) and evaluation.

To run: open the notebook in Google Colab, run all cells in order (`Runtime → Run all`), and authorize Google Drive access when prompted in step 1.

## Dependencies

- tensorflow
- opencv-python
- scikit-image
- scikit-learn
- joblib
- pillow
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- tqdm

All of these are preinstalled in the standard Google Colab runtime except `scikit-image`, which may need to be installed manually on some runtimes with `!pip install scikit-image`.

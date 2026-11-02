# Task 2: Transfer Learning with CNNs
3D Club Induction Task 2026 (Sophomore Batch)

Image classification of flowers (5 classes) using a CNN pretrained on ImageNet.

## Notebook
`transfer_learning_cnn.ipynb`

## Dataset and architecture
- Dataset: TensorFlow Flowers (`flower_photos`, ~3,670 images: daisy, dandelion, roses, sunflowers, tulips), downloaded automatically. Stratified 70/15/15 train/validation/test split.
- Backbone: MobileNetV2 pretrained on ImageNet. New head: GlobalAveragePooling -> Dropout(0.3) -> Dense(5, softmax).
- Training: (1) frozen backbone, (2) fine-tuning of the last 30 layers at learning rate 1e-5 with BatchNorm kept frozen. A small CNN trained from scratch serves as a reference.
- Evaluation: accuracy, macro precision/recall/F1, per-class report, confusion matrices.

## How to run
A GPU is recommended (Google Colab works).
1. `pip install -r requirements.txt`
2. Run `jupyter notebook` and execute all cells.

## Results
Add your final metrics table, plots and conclusion here.

## Sources
- TensorFlow Flowers dataset (TensorFlow team).
- Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.

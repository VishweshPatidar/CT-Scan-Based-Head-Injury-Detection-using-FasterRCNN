# CT Scan-based Brain Hemorrhage Detection using Faster R-CNN

This repository contains the code and resources for a deep learning-based project aimed at detecting brain hemorrhages from CT scan images using the Faster R-CNN architecture. This project was developed as part of the Minor 1 curriculum for a B.Tech in Computer Science and Engineering.

## Overview
Brain hemorrhages are life-threatening conditions that require immediate attention. This project utilizes Faster R-CNN, a state-of-the-art object detection model, to identify hemorrhage regions in CT scan images. The goal is to assist medical professionals by providing an automated detection system.

## Features
- **Model Architecture**: Faster R-CNN for object detection.
- **Datasets Used**:
  - Roboflow's CT Brain Hemorrhage Dataset
- **Input Image Resolution**: 640x640 pixels.
- **Output**: Detected regions of hemorrhages in CT scan images with bounding boxes and probabilities.

## Dataset
The project uses:
1. **Roboflow CT Brain Hemorrhage Dataset**: Curated dataset available from Roboflow, formatted for object detection tasks.

### Preprocessing Steps
- Images resized to 640x640.
- Label annotations were formatted to be compatible with Faster R-CNN requirements.

## Model
The Faster R-CNN model was selected for its ability to provide high accuracy in object detection tasks. The architecture consists of:
- Region Proposal Network (RPN)
- Feature Pyramid Network (FPN)
- Fast RCNN classifier and regressor for object detection.

## Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/ct-scan-injury-detection.git
cd ct-scan-injury-detection
pip install -r requirements.txt
```

## Usage
1. **Prepare the dataset**:
   - Download the datasets and place them in the `data/` directory.
   - Ensure the annotations are in COCO or Pascal VOC format.

2. **Train the model**:
   ```bash
   python train.py --config configs/faster_rcnn_config.yaml
   ```

3. **Evaluate the model**:
   ```bash
   python evaluate.py --model checkpoints/best_model.pth
   ```

4. **Run inference**:
   ```bash
   python infer.py --image path/to/your/image.png
   ```

## Results
- **Performance Metrics**:
  - Mean Average Precision (mAP): *Add your result*
  - Training Loss: *Add your result*
- **Sample Output**:
  Include a screenshot or example of a detected CT scan image.

## Folder Structure
```
ct-scan-injury-detection/
|
|-- configs/              # Configuration files
|-- data/                 # Datasets
|-- models/               # Model definition files
|-- checkpoints/          # Saved model weights
|-- utils/                # Utility scripts
|-- train.py              # Script for training
|-- evaluate.py           # Script for evaluation
|-- infer.py              # Script for inference
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
```

## Future Work
- Extend the model to classify different types of brain hemorrhages.
- Improve the model's accuracy with additional data augmentation techniques.
- Deploy the model as a web-based application for real-time inference.

## Contributors
- **Vishwesh Patidar** ([GitHub Profile](https://github.com/VishweshPatidar))
- **Abhigyan Shrivastava** ([GitHub Profile](https://github.com/abhiigyan))

## Acknowledgements
- Roboflow for dataset hosting
- Faster R-CNN authors and PyTorch community
- Various research paper available at IEEE Website

## License
This project is licensed under the MIT License. See the LICENSE file for details.

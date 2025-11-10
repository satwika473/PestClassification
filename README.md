# Intelligent Pest Classification

An AI-powered pest classification system that helps identify and provide treatment recommendations for agricultural pests.

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/satwika473/PestClassification.git
cd PestClassification
```

2. Download the model file
- Download `vit_best.pth` from https://drive.google.com/file/d/1zoaGv0626KqpwrVD4v2GjGZNyqYs-oW6/view?usp=sharing
- Place it in the root directory of the project

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

## Features
- Real-time pest detection using Vision Transformer
- Detailed pest information and prevention tips
- Multi-language support for prevention measures
- Responsive agricultural-themed interface

## Model File
The Vision Transformer model file (`vit_best.pth`) is not included in this repository due to size limitations. Please download it from:
(https://drive.google.com/file/d/1zoaGv0626KqpwrVD4v2GjGZNyqYs-oW6/view?usp=sharing)

## Tech Stack
- Python Flask
- PyTorch (Vision Transformer)
- HTML/CSS
- JavaScript
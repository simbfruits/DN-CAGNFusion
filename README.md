DN-CAGNFusion: Multimodal Image Fusion
This project implements a deep learning-based framework for multimodal image fusion. It utilizes Global Luminance Perception (GLP) and Spatial Aware Refinement (SAR) to enhance visible light images, combined with the DNFN model for high-quality fusion of infrared and visible spectra.

Core Architecture
The project consists of three primary modules:

GLPN (Global Luminance Perception): Predicts luminance gain for visible images, enabling global adaptive perception.

SARN (Spatial Aware Refinement): A refinement module that enhances local texture details in visible light images.

DNFN (Deep Nested Fusion Network): Utilizes a deep nested structure to fuse features from infrared images and enhanced visible light images.

Environment Requirements
The project is built on PyTorch. Ensure you have the following installed:

torch

torchvision

Pillow (PIL)

Usage Guide
1. Project Structure
Ensure your root directory includes the necessary pre-trained weight files:
DN-CAGNFusion/
├── checkpoint/enhanced_train/   # Contains LFN and LAN weights
├── DNFN/runs/                   # Contains DNFN weights
└── DNFN/fusion_test_data/       # Test data directory
2. Running the Fusion
Execute the following command to perform image fusion:
python FOSION.py \
    --vis_dir "./DNFN/fusion_test_data/L/vi" \
    --ir_dir "./DNFN/fusion_test_data/L/ir" \
    --result_path "./DNFN/fusion_test_data/L/CAGN" \
    --fusion_weight 0.5
3. Optional Parameters
--save_enhanced: If enabled, saves the intermediate results after GLPN+SARN enhancement.

--fusion_weight: Sets the infrared fusion weight ratio (0.0 - 1.0), default is 0.5.

--debug: Enables debug mode, printing detailed processing flows for the first image.

Technical Highlights
Color Space: Fusion is performed in the YCbCr space. By fusing only the luminance channel (Y) and preserving the chrominance channels (Cb, Cr) from the source visible image, color distortion is effectively minimized.

Deep Supervision: Supports deepsupervision mode to enhance the stability of feature extraction.
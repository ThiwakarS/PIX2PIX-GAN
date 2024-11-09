# SAR Image Colorization using Pix2Pix GAN

This project was developed for Smart India Hackathon to automate the colorization of black and white Synthetic Aperture Radar (SAR) satellite images using a Pix2Pix Generative Adversarial Network (GAN). The system aims to reduce manual colorization work required for geographical visualization.

## Project Overview

The project converts grayscale satellite imagery into colorized versions using deep learning, specifically focusing on:
- Agricultural land
- Barren land
- Grassland
- Urban land

## Technical Architecture

The solution implements a Pix2Pix GAN with:
- A U-Net-based generator
- A convolutional PatchGAN discriminator
- Custom loss functions combining adversarial loss and L1 loss

### Model Details

#### Generator
- U-Net architecture with skip connections
- Encoder-decoder structure
- Input/Output image size: 256x256x3
- Activation: ReLU for encoder, LeakyReLU for decoder
- Batch normalization and dropout for regularization

#### Discriminator
- PatchGAN architecture
- Processes 256x256 images
- Uses LeakyReLU activation
- Implements batch normalization
- Outputs a matrix of real/fake predictions

## Setup and Installation

### Prerequisites
```bash
# Required packages
tensorflow
keras
numpy
matplotlib
pandas
```

### Project Structure
```
├── main.py                 # Main training script
├── pix2pix_GAN.py         # GAN model architecture and training functions
├── dataset/
│   └── agri/             # Dataset directory
│       ├── s1/          # Source images (grayscale)
│       └── s2/          # Target images (colored)
├── Images/               # Generated sample outputs
├── G_model/             # Saved generator models
├── D_model/             # Saved discriminator models
├── Loss_plots/          # Training loss visualizations
└── loss_values.xlsx     # Training loss records
```

## Usage

1. Prepare your dataset:
   - Place grayscale images in `dataset/agri/s1/`
   - Place corresponding colored images in `dataset/agri/s2/`

2. Run the training:
```python
python main.py
```

## Training Process

The model is trained with:
- Image size: 256x256 pixels
- Batch size: 1
- Adam optimizer (learning rate: 0.0002, beta_1: 0.5)
- Loss weights: adversarial loss (5) and L1 loss (100)

Training progress is monitored through:
- Regular sample image generation
- Loss tracking for both generator and discriminator
- Automatic model checkpointing

## Results

The model shows promising results after 5-6 epochs, demonstrating:
- Stable training progression
- Realistic color reproduction
- Preservation of structural details

## Model Output Directories

- `Images/`: Contains sample outputs during training
- `G_model/`: Stores generator model checkpoints
- `D_model/`: Stores discriminator model checkpoints
- `Loss_plots/`: Contains training loss visualizations

## Performance Monitoring

The system includes comprehensive monitoring:
- Real-time loss tracking
- Sample image generation every epoch
- Automatic progress saving
- Loss visualization plots

## Features

- Automated colorization of SAR images
- Support for multiple terrain types
- Real-time training visualization
- Modular architecture for easy modifications
- Comprehensive loss tracking and visualization

## Future Improvements

Potential enhancements could include:
- Multi-scale discrimination
- Attention mechanisms
- Additional data augmentation
- Support for higher resolution images
- Integration with web interface

## License

MIT License

Copyright (c) 2024 ThiwakarS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This project was developed as part of Smart India Hackathon, focusing on reducing manual effort in satellite image processing while maintaining high accuracy in geographical visualization.

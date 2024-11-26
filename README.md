# 3D U-Net Segmentation for Liver Sinusoids

This repository contains the implementation of a 3D U-Net model designed to segment liver sinusoids networks from microscopy imaging data. The project includes scripts for data preprocessing, model training, inference, and visualization, making it easier for researchers to work on liver sinusoid segmentation tasks.


## Requirements

- Python 3.9
- PyTorch
- NumPy
- SciPy
- scikit-image
- tifffile
- matplotlib
- mayavi
- PyQt5

## Compatibility

- **OS**: The code has been tested on Linux and Windows.
- **Hardware**: Note that the 3D PyTorch library is currently not available for Apple M1/M2 chips. Running on these platforms is not yet supported (nov 2024).

## Setup

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/akolossa/3d_unet_segmentation_sinusoids.git
   cd 3d_unet_segmentation_sinusoids
   ```

2. **Install the required packages:**
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

Use the `dataset_augmented.py` script to preprocess the input data and prepare it for training. This script includes data augmentation techniques and applies basic 3D transformations using TorchIO.

```bash
python dataset_augmented.py
```

### Training

Train the 3D U-Net model using the `train.py` script. You can adjust configuration parameters, such as learning rate, batch size, and paths, by modifying the `config.py` file.

```bash
python train.py
```

### Inference

To perform inference on new data, use the `inference.py` script. This will generate segmentation predictions for your liver sinusoid images.

```bash
python inference.py
```

### Visualization

Visualize the input, ground truth, and predicted segmentation results using the `visualise_images.py` script.

```bash
python visualise_images.py
```

### Skeletonization

Use the `skeletonisation.py` script to skeletonize the segmented images. Skeletonization reduces structures to a simplified representation that preserves essential geometrical properties.

```bash
python skeletonisation.py
```

### Path Extrusion

Perform path extrusion on the skeletonized images using the `path_extrusion.py` script. Path extrusion can be useful for further analysis of the structure, such as identifying the main flow paths through the sinusoidal network.

```bash
python path_extrusion.py
```

## Configuration

All configuration settings for the project can be found in `config.py`. This includes paths to training data, hyperparameters for model training, and various model-specific settings. Modify these settings as needed to suit your specific experimental setup.

## Contributing

We welcome contributions to this project! If you find any issues, feel free to open an issue on GitHub. Pull requests for new features, bug fixes, or improvements are also appreciated.

## Acknowledgements

This project is based on the 3D U-Net architecture for medical image segmentation. Special thanks to the contributors of the open-source libraries used, including PyTorch, scikit-image, and TorchIO, among others.

## Contact

For questions, concerns, or to request a pretrained model (with 80% accuracy), please reach out:

- Email: [arawa@hotmail.it](mailto:arawa@hotmail.it) or [arawa.kolossa@ru.nl](mailto:arawa.kolossa@ru.nl)

## License

This repository is licensed under the MIT License. 


MIT License

Copyright (c) 2024 Arawa Kolossa

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



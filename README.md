# Network
Neural Network training and running c/c++ 

**Neural Network for Sperm Abnormality Detection**

This is a simple neural network implementation in C++ designed to detect and classify abnormalities in human sperm cells from grayscale images. It performs multilabel classification across four categories: acrosome, head, vacuole, and tail abnormalities.

_Features_

- Fully custom neural network implementation (no deep learning frameworks)
- 
- Multilabel binary classification using sigmoid + binary cross-entropy loss
- 
- Batch training with learning rate decay and L2 regularization
- 
- OpenBLAS acceleration for matrix operations
- 
- Per-sample input normalization for improved stability
- 
- Accuracy metrics per label and overall

**Dataset**

This project uses the MHMSA dataset, which contains labeled grayscale images of sperm cells with multiple abnormalities. Labels are provided as binary vectors for each image.

**Build Instructions**
Make sure you have:

- A C++ compiler (e.g. MinGW on Windows)
- 
- [OpenBLAS](https://www.openblas.net/) installed
- 
- [stb_image](https://github.com/nothings/stb) headers for PNG loading
- 
- You can build the project using Code::Blocks or a standard Makefile with OpenBLAS linked.

**Training**
Training is performed over a set number of epochs with options for:

- Batch size
- 
- Learning rate
- 
- Regularization factor
- 
- Early stopping (WIP)

**License**
This project is released under the MIT License.

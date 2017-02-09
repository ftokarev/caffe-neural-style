# Neural Style

## Setup

1. Build [a modified version of Caffe](https://github.com/ftokarev/caffe/tree/neural-style), which provides:
    - Gram layer
    - Total Variation Loss layer
    - L-BFGS solver

2. Install python dependencies:

    ```pip install -r requirements.txt```

3. Get the VGG model:

    ```cd vgg; ./get_model.sh```

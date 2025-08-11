# Consolidated List of CIFAR-Format & Few-Shot Learning Datasets

This document provides a comprehensive list of image classification datasets that are either in the CIFAR format (small, 32x32 color images) or are commonly used in similar research contexts, such as few-shot learning. Each entry includes a description, its origin, and direct download links or code snippets for easy access.

## I. Core CIFAR & Direct Variants

These are the official CIFAR datasets and their direct derivatives designed for specific tasks like few-shot learning or robustness testing.

### CIFAR-10 & CIFAR-100
- **Description**: The foundational datasets. Both contain 60,000 32x32 color images. CIFAR-10 has 10 classes, and CIFAR-100 has 100 classes. They are standard benchmarks for image classification.
- **Source**: University of Toronto (Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton).
- **Access**:
    - **Direct Download**:
        - `wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
        - `wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
    - **Programmatic (PyTorch)**:
        ```python
        from torchvision import datasets
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR100(root='./data', train=True, download=True)
        ```

### CIFAR-FS (Few-Shot)
- **Description**: A few-shot learning benchmark derived from CIFAR-100. It uses all 100 classes but splits them into 64 for training, 16 for validation, and 20 for testing. The splits are designed to be challenging.
- **Source**: Derived from CIFAR-100.
- **Access**:
    - **Programmatic (learn2learn)**:
        ```python
        import learn2learn as l2l
        dataset = l2l.vision.datasets.CIFARFS(root='./data', mode='train', download=True)
        ```

### FC100 (Few-Shot Classes 100)
- **Description**: A more challenging few-shot dataset also based on CIFAR-100. It groups the 100 fine-grained classes into 20 coarse-grained superclasses and splits these superclasses to minimize semantic overlap between training, validation, and test sets (60 train / 20 val / 20 test classes).
- **Source**: Derived from CIFAR-100.
- **Access**:
    - **Programmatic (learn2learn)**:
        ```python
        import learn2learn as l2l
        dataset = l2l.vision.datasets.FC100(root='./data', mode='train', download=True)
        ```

### CIFAR-10-C & CIFAR-100-C (Corruption)
- **Description**: Test sets for measuring a model's robustness. They apply 19 different types of algorithmically generated corruptions (e.g., noise, blur, weather) to the original CIFAR test images.
- **Source**: Dan Hendrycks & Thomas Dietterich.
- **Access**:
    - **Homepage**: [GitHub - hendrycks/robustness](https://github.com/hendrycks/robustness)
    - **Direct Download**:
        - `wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar`
        - `wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar`

### CIFAR-10.1
- **Description**: A new, re-collected test set for CIFAR-10 to measure generalization, containing 2,000 new images that avoid duplicates from the original training set.
- **Source**: Ludwig Schmidt, Shibani Santurkar, et al.
- **Access**:
    - **Homepage**: [GitHub - modestyachts/CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1)
    - **Programmatic (TensorFlow Datasets)**:
        ```python
        import tensorflow_datasets as tfds
        ds = tfds.load('cifar10_1', split='test')
        ```

### CIFAR-10H
- **Description**: The CIFAR-10 test set with additional soft labels from human annotators, capturing label uncertainty. Useful for training models that can represent ambiguity.
- **Source**: Jonas C. Peterson, et al.
- **Access**:
    - **Direct Download (via Git)**:
        ```bash
        git clone https://github.com/jcpeterson/cifar-10h.git
        ```

## II. CIFAR-Style Benchmarks

These datasets are very similar in format or common usage to CIFAR and are often used as drop-in replacements or for complementary experiments.

### CINIC-10
- **Description**: An extended version of CIFAR-10, containing 270,000 images (90k for train, val, and test each). It combines images from CIFAR-10 and downsampled ImageNet, maintaining the same 10 classes and 32x32 format.
- **Source**: University of Edinburgh.
- **Access**:
    - **Direct Download**:
        `wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz`

### SVHN (Street View House Numbers)
- **Description**: Images of house numbers collected from Google Street View. The cropped format contains over 600,000 32x32 color images across 10 classes (digits 0-9).
- **Source**: Stanford University (Yuval Netzer et al.).
- **Access**:
    - **Direct Download**:
        - `wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat`
        - `wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat`
    - **Programmatic (PyTorch)**:
        ```python
        from torchvision import datasets
        datasets.SVHN(root='./data', split='train', download=True)
        ```

### Fashion-MNIST
- **Description**: A drop-in replacement for the original MNIST dataset, featuring 70,000 28x28 grayscale images of 10 clothing categories.
- **Source**: Zalando Research.
- **Access**:
    - **Homepage**: [GitHub - zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
    - **Direct Download**: [Download Links](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)
    - **Programmatic (PyTorch)**:
        ```python
        from torchvision import datasets
        datasets.FashionMNIST(root='./data', train=True, download=True)
        ```

### STL-10
- **Description**: Inspired by CIFAR-10 but designed for unsupervised and semi-supervised learning. It contains 5,000 labeled 96x96 training images (10 classes), 8,000 labeled test images, and a large set of 100,000 unlabeled images.
- **Source**: Stanford University (Adam Coates et al.).
- **Access**:
    - **Direct Download**: `wget https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz`
    - **Programmatic (PyTorch)**:
        ```python
        from torchvision import datasets
        datasets.STL10(root='./data', split='train', download=True)
        ```

## III. Scaled & Few-Shot Benchmarks

These datasets are often larger than CIFAR-10 or are structured specifically for few-shot and meta-learning research.

### Tiny ImageNet
- **Description**: A subset of ImageNet with 200 classes. It includes 100,000 64x64 color images for training, plus 10,000 for validation and 10,000 for testing. A popular step-up in complexity from CIFAR.
- **Source**: Stanford CS231N course.
- **Access**:
    - **Direct Download**: `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`

### miniImageNet
- **Description**: A standard benchmark for few-shot learning, curated from ImageNet. It contains 100 classes with 600 samples each, typically resized to 84x84. The standard splits are 64/16/20 for train/val/test classes.
- **Source**: Originally proposed by Vinyals et al. (2016).
- **Access**:
    - **Kaggle**: [miniImageNet Dataset](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) (Requires free Kaggle account)
    - **Programmatic (learn2learn)**:
        ```python
        import learn2learn as l2l
        dataset = l2l.vision.datasets.MiniImageNet(root='./data', mode='train', download=True)
        ```

### Omniglot
- **Description**: A dataset for "one-shot" learning, often called the "transpose of MNIST." It contains 1,623 different handwritten characters from 50 alphabets, with only 20 examples for each character.
- **Source**: Brenden Lake, et al.
- **Access**:
    - **Homepage**: [GitHub - brendenlake/omniglot](https://github.com/brendenlake/omniglot)
    - **Direct Download**:
        - `wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip`
        - `wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip`

### Downsampled ImageNet
- **Description**: The full ImageNet dataset downsampled to 32x32 or 64x64 resolution, making it tractable to train models without complex data pipelines while still providing a large-scale, diverse dataset.
- **Source**: Originally from van den Oord et al. (2016).
- **Access**:
    - **Programmatic (TensorFlow Datasets)**:
        ```python
        import tensorflow_datasets as tfds
        ds32 = tfds.load('downsampled_imagenet/32x32', split='train')
        ds64 = tfds.load('downsampled_imagenet/64x64', split='train')
        ```

### Caltech-101
- **Description**: An object recognition dataset containing ~9,000 images across 101 object categories (plus a background class). Image sizes are variable but are often resized for classification tasks.
- **Source**: California Institute of Technology (Fei-Fei Li et al.).
- **Access**:
    - **Homepage**: [Caltech 101 Dataset](https://data.caltech.edu/records/20086)

### Meta-Album
- **Description**: A meta-dataset of 40+ smaller, CIFAR-like datasets designed for robust few-shot learning evaluation across diverse domains. Each sub-dataset has around 20+ classes with ~40 images per class.
- **Source**: Ihsan Ullah, et al.
- **Access**:
    - **Homepage**: [Meta-Album Project](https://meta-album.github.io/)
    - **arXiv Paper**: [Meta-Album: Multi-domain Meta-Dataset](https://arxiv.org/abs/2302.08909)
# Weakly Supervised Object Detection
My implementations of a variety of algorithms and models for the weakly supervised object detection problem, based on facebookresearch/detectron2

## Requirements
- Python 3
- Latest Pytorch (1.3.1 as of last readme update)
- Latest Torchvision (0.4.2 as of last readme update)
- Detectron2

## Installation Instructions
We follow similar installation as that of Detectron2.

1. Clone this repository to `$WORK_DIR`.
    ```
    cd $WORK_DIR; git clone https://github.com/bradezard131/wsod.git
    ```
2. Clone Detectron2 to `$WORK_DIR`.
    ```
    cd $WORK_DIR; git clone https://github.com/facebookresearch/detectron2.git
    ```
3. Make Conda Environment.
    ```
    conda create -n wsod python=3.7 -y; conda activate wsod
    ```
4. Install Pytorch
    ```
    conda install pytorch=1.3 torchvision cudatoolkit=10.1 -c pytorch
    ```
5. Install cocoapi
    ```
    pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ```
6. Install Detectron2
    ```
    cd $WORK_DIR/detectron2; pip install -e .
    ```
7. Install scikit-learn
    ```
    pip install scikit-learn
    ```
    

## Models implemented in this repository
Below are the models currently implemented in this repository, as well as their performance using VGG16 as a backbone.
- [Weakly Supervised Deep Detection Networks (Bilen et. al., 2015)](https://arxiv.org/abs/1511.02853) (26.17, no spatial regularisation)
- [Multiple Instance Detection Network with Online Instance Classifier Refinement (Tang et. al., 2017)](https://arxiv.org/abs/1704.00138) (40.83)
- [PCL: Proposal Cluster Learning for Weakly Supervised Object Detection (Tang et. al., 2018)](https://arxiv.org/abs/1807.03342) (44.21)

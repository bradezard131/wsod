# Weakly Supervised Object Detection
My implementations of a variety of algorithms and models for the weakly supervised object detection problem, based on facebookresearch/detectron2

## Requirements
- Python 3
- Latest Pytorch (1.3.1 as of last readme update)
- Latest Torchvision (0.4.2 as of last readme update)
- Detectron2

## Models implemented in this repository
Below are the models currently implemented in this repository, as well as their performance using VGG16 as a backbone.
- [Weakly Supervised Deep Detection Networks (Bilen et. al., 2015)](https://arxiv.org/abs/1511.02853) (26.17, no spatial regularisation)
- [Multiple Instance Detection Network with Online Instance Classifier Refinement (Tang et. al., 2017)](https://arxiv.org/abs/1704.00138) (40.83)
- [PCL: Proposal Cluster Learning for Weakly Supervised Object Detection (Tang et. al., 2018)](https://arxiv.org/abs/1807.03342) (44.21)

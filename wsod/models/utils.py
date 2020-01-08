import torch
from torch import nn
#from detectron2.layers import batched_nms
from torchvision.ops.boxes import batched_nms


@torch.no_grad()
def orthogonal_init(layers, mean=0.0, std=0.01):
    k = len(layers)
    ou_f = layers[0].out_features
    in_f = layers[0].in_features
    random = torch.randn((ou_f, in_f, k)) * std + mean
    q, r = torch.qr(random, some=True)

    for detector, init in zip(layers, q.permute(2, 0, 1)):
        detector.weight.data.copy_(init)
        nn.init.zeros_(detector.bias)


@torch.no_grad()
def filter_predictions(scores, rois, nms_threshold, score_threshold):
    rois = rois.to(scores.device)
    idxs, cls_ids = (scores > score_threshold).nonzero().T
    cls_scores = scores[idxs, cls_ids]
    boxes = rois[idxs]
    keep = batched_nms(boxes, cls_scores, cls_ids, nms_threshold)
    return boxes[keep], cls_scores[keep], cls_ids[keep]


@torch.no_grad()
def load_weights(convs, fc, pretrained):
    m = torch.load(pretrained)
    for model_param, pretrained_param in zip(list(convs.parameters()) + list(fc.parameters()), 
                                             m.parameters()):
        model_param.weight.copy_(pretrained_param.weight)
        model_param.bias.copy_(pretrained_param.bias)


def get_conv_scale(convs):
    """
    Determines the downscaling performed by a sequence of convolutional and pooling layers
    """
    scale = 1.
    for c in convs:
        stride = getattr(c, 'stride', 1.)
        scale /= stride if isinstance(stride, (int, float)) else stride[0]
    return scale


def get_out_features(fc):
    """
    Determines the size of the output from a sequence of fully connected layers
    """
    i = -1
    while i < 0:  # will be set to out features to exit
        i = getattr(fc[i], 'out_features', i-1)
    return i


def freeze_convs(convs, k):
    """
    Freezes `k` conv layers
    """
    i = 0
    while k > 0:
        if isinstance(convs[i], nn.Conv2d):
            k -= 1
            for p in convs[i].parameters():
                p.requires_grad = False
        i += 1


def extract_data(element):
    image = element['image']
    
    instances = element.get('instances')
    if instances:
        gt_boxes = instances.gt_boxes
        gt_classes = instances.gt_classes
    else:
        gt_boxes = torch.zeros((0, 4), dtype=torch.float)
        gt_classes = torch.zeros((0,), dtype=torch.long)

    rois = element['proposals'].proposal_boxes.tensor
    
    return image, rois, gt_classes, gt_boxes
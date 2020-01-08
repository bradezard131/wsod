import torch
from torch import nn
from torchvision import models, ops


__backbones = {
    'vgg16': models.vgg16
}


class DiscoveryModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, rois, targets=None):
        x = self.lin(x)
        n, c = x.shape
        x = x.view(-1).softmax(0).view(n,c)
        boxes = rois
        cliques = []
        while x.size(0) > 0:
            scores, classes = x.max(1)
            top_score, top_idx = scores.max(0)
            box = boxes[top_idx]
            overlaps = ops.box_iou(box.view(1,4), boxes)
            clique_ids = (overlaps > 0.7).nonzero()[:,1]
            cliques.append(clique_ids)
            keep = (overlaps <= 0.7).nonzero()[:,1]
            boxes = boxes[keep, keep]
            x = x[keep]

        if self.training:
            assert targets is not None, "Must have targets for training"
            loss = 0
            for clique in cliques:
                scores = x[clique]
                weights = 1. / len(cliques) * (scores / scores.sum(1)).sum(0)
                clique_scores = scores.sum(0)
                loss -= ((weights * clique_scores) * targets).log().sum()
            loss -= ((1 - targets) * (1 - x).log()).sum()
            return cliques, x, loss
        else:
            return cliques, x, 0


class LocalisationModule(nn.Module):
    
        


class MinEntropyLatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = __backbones['vgg16'](pretrained=True)
        
        self.convs = base.features[:-1]
        self.pooler = ops.RoIPool((7,7), 1./16.)
        self.fc = base.classifier[:-1]
        
        self.loc = LocalisationModule()
        self.dis = DiscoveryModule(4096, 20)

    def predict_on_batch(self, images, rois):
        x = self.convs(x)
        x = self.pooler(x, rois)
        x = x.flatten(1)
        x = self.fc(x)
        
        for rois_per_image in rois:
            n = rois.size(0)
            x_i = x[:n]
            x = x[n:]
            cliques, g, loss = self.dis(x_i, rois_per_image)
            l = self.loc(x, g)
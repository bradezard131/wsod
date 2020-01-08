import torch
from torch import nn
from torchvision import models, ops


class InstanceSelector(nn.Module):
    def __init__(self, in_features, out_features, epochs=20, schedule='log'):
        self.fc = nn.Linear(in_features, out_features)
        
    def partition(self, x, rois, lamb):
        pred_scores, pred_classes = x.max(dim=1)
        sort_idx = pred_scores.argsort(descending=True)
        subsets = []
        while sort_idx.size(0) > 0:
            idx = sort_idx[0]
            
            overlaps = ops.box_iou(rois[idx].view(1,4), rois)
            subset = (overlaps >= lamb).nonzero()[:,1]
            keep = (overlaps < lamb).nonzero()[:,1]
            
            subsets.append(sort_idx[subset])
            sort_idx = sort_idx[keep]
            rois = rois[keep]
        return subsets

    def forward(self, x, rois, lamb=1, targets=None):
        x = self.fc(x)
        partitions = self.partition(x, rois, lamb)
        
        if self.training:
            # Convert 0,1 targets to -1,1
            targets[targets == 0] = -1
            losses = torch.zeros(x.size(-1), dtype=x.dtype, device=x.device)
            for partition in partitions:
                scores = x[partition].mean(dim=0)
                partition_losses = 1 - targets * scores
                losses = torch.max(losses, partition_losses)
            loss = losses.mean()
        else:
            loss = 0
        return partitions, loss


class DetectorEstimator(nn.Module):
    def __init__(self, in_features, out_features):
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, rois, lamb=1, target=None):
        x = self.fc(x).softmax(-1)
        
        if self.training:
            score = x[:,target]
            top = score.argmax()
            overlaps = ops.box_iou(rois[top].view(1,4), rois)
            fg = overlaps >= 1 - lamb / 2
            bg = overlaps < lamb / 2
            # ignore lamb / 2 <= x < 1-lamb/2
            loss = 0
            loss -= score[fg].log().sum()
            loss -= (1-score[bg]).log().sum()
            return loss
        else:
            return x


class ContinuationMIL(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base = models.vgg16(pretrained=True)
        self.convs = base.features[:-1]
        self.pooler = ops.RoIPool((7,7), 1./16.)
        self.fc = base.classifier[:-1]
        
        self.instance_selector = InstanceSelector(in_features, 20)
        self.detector_estimator = DetectorEstimator(in_features, 21)
        
        self.lambda = 0.0
        self.schedule = self.build_schedule(epochs, schedule)
        
    def build_schedule(self, epochs, schedule):
        if schedule == 'linear':
            return torch.linspace(0, 1, epochs)
        elif schedule == 'log':
            return torch.logspace(0,1,20)/10 - torch.linspace(0.1, 0, 20)
        elif schedule == 'sigmoid':
            return torch.linspace(-13,13,epochs).sigmoid()
        elif schedule == 'exp':
            backward = 1 - self.build_schedule(self, epochs, 'log')
            return torch.tensor([backward[i] for i in range(-1, -(len(backward)+1), -1)])
        elif schedule == 'piecewise':
            return torch.tensor(
                [0.2] * epochs // 5 +
                [0.4] * epochs // 5 +
                [0.6] * epochs // 5 +
                [0.8] * epochs // 5 +
                [1.0] * epochs - 4 * (epochs // 5)
            )
    
    def step(self):
        self.lambda = self.schedule[0]
        self.schedule = self.schedule[1:]
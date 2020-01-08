import torch
from torch import nn


class MultipleMidnHead(nn.Module):
    def __init__(self, in_features, out_features, t_cls=1., t_det=1., k_cls=1, k_det=1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.classifiers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(k_cls)])
        self.detectors = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(k_det)])
        self.t_cls = t_cls
        self.t_det = t_det
        self.k = k_cls * k_det

    def forward(self, x):
        result = []
        for cls in self.classifiers:
            c = (cls(x) / self.t_cls).softmax(1)
            for det in self.detectors:
                d = (det(x) / self.t_det).softmax(0)
                result.append(c * d)
        return result


class RefinementHeads(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.refinements = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(k)])
        self.k = k

    def forward(self, x):
        result = []
        for refinement in self.refinements:
            result.append(refinement(x))
        return result 
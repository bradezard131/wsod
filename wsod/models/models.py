import numpy as np
import torch
from torch import nn
from torchvision import models as M, ops

from detectron2.data import transforms as T
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes

from . import heads, utils
from .losses import LOSS_FUNCTIONS
from .backbones.vggm import vggm_1024

from typing import List


_backbones = {
    'alexnet': lambda p: extract_components(M.alexnet, p),
    'vgg16': lambda p: extract_components(M.vgg16, p),
    'vggm': lambda p: extract_components(vggm_1024, p),
}


def extract_components(model_fn, pretrained=False):
    model = model_fn(pretrained)
    convs = model.features[:-1]
    fc    = model.classifier[:-1]
    return convs, fc


def dilate_convs(convs):
    i = -1
    while not isinstance(convs[i], nn.MaxPool2d):
        if isinstance(convs[i], nn.Conv2d):
            convs[i].dilation = (2, 2)
            convs[i].padding = (2, 2)
        i -= 1
    del convs[i]
    return convs


@META_ARCH_REGISTRY.register()
class GeneralisedMIL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg.MODEL.DEVICE
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        # Test mode details
        self.test_nms_threshold = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_score_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.test_out_layers = cfg.MODEL.PREDICTION_LAYERS

        # Normalization details
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1,3,1,1).to(self.device)
        self.pixel_std  = torch.tensor(cfg.MODEL.PIXEL_STD).view(1,3,1,1).to(self.device)
        
        # Set up the model base
        backbone_name = cfg.MODEL.BACKBONE.NAME
        dilated = backbone_name.endswith('_dilated')
        backbone_name = backbone_name[:-len('_dilated')] if dilated else backbone_name
        
        pretrained = cfg.MODEL.BACKBONE.WEIGHTS
        convs, fc = _backbones[backbone_name](pretrained=='imagenet')
        if pretrained not in ['imagenet', '']:
            utils.load_weights(convs, fc, pretrained)

        if dilated:
            convs = dilate_convs(convs)

        utils.freeze_convs(convs, cfg.MODEL.BACKBONE.FREEZE_CONVS)
        self.convs = convs
        self.fc = fc

        # Set up the pooling layer
        scale = utils.get_conv_scale(convs)
        res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        if pool_type.lower() == 'roipool':
            self.pooler = ops.RoIPool((res, res), scale)
        else:
            raise NotImplementedError(f'Pooler type {pool_type} not implemented')

        # Set up the heads
        fc_features = utils.get_out_features(fc)
        nc, nd = cfg.MODEL.MIDN_HEAD.NUM_CLASSIFIER, cfg.MODEL.MIDN_HEAD.NUM_DETECTOR
        if nc > 0 and nd > 0:
            self.midn = heads.MultipleMidnHead(
                in_features=fc_features, 
                out_features=self.num_classes, 
                t_cls=cfg.MODEL.MIDN_HEAD.CLASSIFIER_TEMP, 
                t_det=cfg.MODEL.MIDN_HEAD.DETECTOR_TEMP, 
                k_cls=nc,
                k_det=nd
            )

        nr = cfg.MODEL.REFINEMENT_HEAD.K
        if nr > 0:
            self.refinement = heads.RefinementHeads(
                in_features=fc_features,
                out_features=self.num_classes+1,  #BG Class
                k=3
            )

        if cfg.TEST.AUG.ENABLED:
            self.tta = self._init_tta_fn(cfg)
        else:
            self.tta = lambda x: x

        self.build_loss = LOSS_FUNCTIONS[cfg.MODEL.LOSS_FN]
        self.init_layers()
    
    @torch.no_grad()
    def init_layers(self):
        params = list(self.midn.classifiers.named_parameters())
        if hasattr(self, 'refinement'):
            params += list(self.refinement.named_parameters())

        if len(self.midn.detectors) > 1:
            utils.orthogonal_init(self.midn.detectors)
        else:
            params += list(self.midn.detectors.named_parameters())

        for k, v in params:
            if 'bias' in k:
                nn.init.zeros_(v)
            else:
                nn.init.normal_(v, mean=0.0, std=0.01)
    
    def hack(self):
        for p_source, p_dest in zip(torch.load('/home/Deep_Learner/work/cleaned/outputs/oicr_vgg_dilated/model_final.pth')['model'].values(),
                                    self.parameters()):
            p_dest.copy_(p_source)

    def to(self, device):
        self.device = device
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)
        return super().to(device)

    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def predict_on_example(self, image:torch.Tensor, rois:List[torch.Tensor]) -> List[List[torch.Tensor]]:
        x = self.normalize(image)
        x = self.convs(x)
        x = self.pooler(x, [r.type(x.dtype) for r in rois])
        x = x.flatten(1)
        x = self.fc(x)
        r = self.refinement(x) if hasattr(self, 'refinement') else []
        
        outputs = []
        for rois_per_image in rois:
            n = rois_per_image.size(0)
            x_i = x[:n]
            r_i = [tmp[:n] for tmp in r]
            x = x[n:]
            r = [tmp[n:] for tmp in r]
            m = self.midn(x_i) if hasattr(self, 'midn') and (self.training or not hasattr(self, 'refinement')) else []
            outputs.append(m + r_i)
        return outputs
    
    def _init_tta_fn(self, cfg):
        max_size = cfg.TEST.AUG.MAX_SIZE
        size_gens = [T.ResizeShortestEdge(sz, max_size, 'choice') for sz in cfg.TEST.AUG.MIN_SIZES]
        flip = T.RandomFlip(1.0)
        
        def tta_fn(image, rois):
            image = image.permute(1, 2, 0).to('cpu').numpy()
            dtype = image.dtype
            image = image.astype(np.uint8)
            
            out_images, out_rois = [], []
            for tfm_gen in size_gens:
                resized_image, tfm = T.apply_transform_gens([tfm_gen], image)
                resized_rois = tfm.transforms[0].apply_box(rois.to('cpu').numpy())
                
                if cfg.TEST.AUG.FLIP:
                    flipped_image, tfm = T.apply_transform_gens([flip], resized_image)
                    flipped_rois = tfm.transforms[0].apply_box(resized_rois)
                    
                    img_batch = torch.stack([
                        torch.from_numpy(resized_image.astype(dtype)).permute(2,0,1),
                        torch.from_numpy(flipped_image.astype(dtype)).permute(2,0,1)
                    ])
                    roi_batch = [
                        torch.from_numpy(resized_rois), 
                        torch.from_numpy(flipped_rois)
                    ]
                else:
                    img_batch = torch.from_numpy(resized_image.astype(dtype)).permute(2,0,1).unsqueeze(0)
                    roi_batch = [torch.from_numpy(resized_rois),]
                out_images.append(img_batch)
                out_rois.append(roi_batch)
            return out_images, out_rois

        return tta_fn

    def forward(self, batch, use_gt=False):
        losses = {}
        batch_predictions = []
        bs = len(batch)
        for element in batch:
            image, rois, gt_classes, gt_boxes = utils.extract_data(element)
            
            if self.training:
                predictions = self.predict_on_example(image.unsqueeze(0).to(self.device), [rois.to(self.device)])
                image_labels = torch.zeros((1, self.num_classes,), dtype=image.dtype, device=self.device)
                image_labels[0, gt_classes.unique()] = 1.
                for prediction in predictions:
                    loss = self.build_loss(prediction, rois, image_labels, gt_boxes=gt_boxes, gt_classes=gt_classes)
                    for k, v in loss.items():
                        v = v.float()
                        running_total = losses.setdefault(k, torch.zeros_like(v))
                        losses[k] = running_total + (v / bs)
            else:
                aug_images, aug_rois = self.tta(image, rois)
                scores = None
                for batch_images, batch_rois in zip(aug_images, aug_rois):
                    predictions = self.predict_on_example(batch_images.to(self.device), [r.to(self.device) for r in batch_rois])
                    for prediction in predictions:
                        if hasattr(self, 'refinement'):
                            p = sum([pred.softmax(-1)[:,1:] for pred in prediction])
                        else:
                            p = sum(prediction)
                        if scores is None:
                            scores = p
                        else:
                            scores += p
                boxes, scores, classes = utils.filter_predictions(scores, rois, self.test_nms_threshold, self.test_score_threshold)
                instances = Instances((element['height'], element['width']))
                instances.scores = scores[:self.test_max_detections_per_image]
                instances.pred_classes = classes[:self.test_max_detections_per_image]
                instances.pred_boxes = Boxes(boxes[:self.test_max_detections_per_image])
                batch_predictions.append({
                    'instances': instances
                })
        return losses if self.training else batch_predictions
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops
from sklearn.cluster import KMeans

from typing import Tuple, Dict, List


EPS = 1e-12


def weighted_softmax_with_loss(score:torch.Tensor, labels:torch.Tensor, weights:torch.Tensor) -> torch.Tensor:
    loss = -weights * F.log_softmax(score, dim=-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    valid_sum = weights.gt(EPS).float().sum()
    if valid_sum < EPS:
        return loss.sum() / loss.numel() 
    else:
        return loss.sum() / valid_sum


@torch.no_grad()
def oicr_label(boxes:torch.Tensor, cls_prob:torch.Tensor, image_labels:torch.Tensor, fg_thresh:float=0.5, bg_thresh:float=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes = boxes.to(cls_prob.device)
    cls_prob = (cls_prob if cls_prob.size(-1) == image_labels.size(-1) else cls_prob[..., 1:]).clone()
    gt_boxes = []
    gt_classes = torch.jit.annotate(List[int], [])
    gt_scores = torch.jit.annotate(List[float], [])
    for i in image_labels.nonzero()[:,1]:
        max_index = cls_prob[:,i].argmax(dim=0)
        gt_boxes.append(boxes[max_index])
        gt_classes.append(int(i)+1)
        gt_scores.append(float(cls_prob[max_index, i]))
        cls_prob[max_index] = 0
    max_overlaps, gt_assignment = ops.box_iou(boxes, torch.stack(gt_boxes)).max(dim=1)

    pseudo_labels = torch.gather(torch.tensor(gt_classes, dtype=torch.long, device=cls_prob.device), 0, gt_assignment)
    pseudo_labels[max_overlaps <= fg_thresh] = 0
    weights = torch.gather(torch.tensor(gt_scores, device=cls_prob.device), 0, gt_assignment)
    weights[max_overlaps < bg_thresh] = 0

    return pseudo_labels.detach(), weights.detach()


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    dev = probs.device
    kmeans = KMeans(n_clusters=5).fit(probs.cpu().numpy())
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return torch.from_numpy(index).to(dev)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    dev = cls_prob.device
    gt_boxes = torch.zeros((0, 4), dtype=boxes.dtype, device=dev)
    gt_classes = torch.zeros((0, 1), dtype=torch.long, device=dev)
    gt_scores = torch.zeros((0, 1), dtype=cls_prob.dtype, device=dev)
    for i in im_labels.nonzero()[:,1]:
        cls_prob_tmp = cls_prob[:, i]
        idxs = (cls_prob_tmp >= 0).nonzero()[:,0]
        idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
        idxs = idxs[idxs_tmp]
        boxes_tmp = boxes[idxs, :]
        cls_prob_tmp = cls_prob_tmp[idxs]

        graph = (ops.box_iou(boxes_tmp, boxes_tmp) > 0.4).float()

        keep_idxs = []
        gt_scores_tmp = []
        count = cls_prob_tmp.size(0)
        while True:
            order = graph.sum(dim=1).argsort(descending=True)
            tmp = order[0]
            keep_idxs.append(tmp)
            inds = (graph[tmp, :] > 0).nonzero()[:,0]
            gt_scores_tmp.append(cls_prob_tmp[inds].max())

            graph[:, inds] = 0
            graph[inds, :] = 0
            count = count - len(inds)
            if count <= 5:
                break
        
        gt_boxes_tmp = boxes_tmp[keep_idxs, :].view(-1, 4).to(dev)
        gt_scores_tmp = torch.tensor(gt_scores_tmp, device=dev)

        keep_idxs_new = torch.from_numpy((gt_scores_tmp.argsort().to('cpu').numpy()[-1:(-1 - min(len(gt_scores_tmp), 5)):-1]).copy()).to(dev)

        gt_boxes = torch.cat((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
        gt_scores = torch.cat((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
        gt_classes = torch.cat((gt_classes, (i + 1) * torch.ones((len(keep_idxs_new), 1), dtype=torch.long, device=dev)))

        # If a proposal is chosen as a cluster center,
        # we simply delete a proposal from the candidata proposal pool,
        # because we found that the results of different strategies are similar and this strategy is more efficient
        another_tmp = idxs.to('cpu')[torch.tensor(keep_idxs)][keep_idxs_new.to('cpu')].numpy()
        cls_prob = torch.from_numpy(np.delete(cls_prob.to('cpu').numpy(), another_tmp, axis=0)).to(dev)
        boxes = torch.from_numpy(np.delete(boxes.to('cpu').numpy(), another_tmp, axis=0)).to(dev)

    proposals = {'gt_boxes' : gt_boxes.to(dev),
                 'gt_classes': gt_classes.to(dev),
                 'gt_scores': gt_scores.to(dev)}

    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = ops.box_iou(all_rois.to(gt_boxes.device), gt_boxes)
    max_overlaps, gt_assignment = overlaps.max(dim=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = (max_overlaps >= 0.5).nonzero()[:,0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = (max_overlaps < 0.5).nonzero()[:,0]

    ig_inds = (max_overlaps < 0.1).nonzero()[:,0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = torch.zeros(gt_boxes.shape[0], dtype=cls_prob.dtype, device=cls_prob.device)
    pc_probs = torch.zeros(gt_boxes.shape[0], dtype=cls_prob.dtype, device=cls_prob.device)
    pc_labels = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=cls_prob.device)
    pc_count = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=cls_prob.device)

    for i in range(gt_boxes.shape[0]):
        po_index = (gt_assignment == i).nonzero()[:,0]
        img_cls_loss_weights[i] = torch.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = (cls_prob[po_index, pc_labels[i]]).mean()

    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights


@torch.no_grad()
def pcl_label(boxes:torch.Tensor, cls_prob:torch.Tensor, im_labels:torch.Tensor, cls_prob_new:torch.Tensor):
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    cls_prob = cls_prob.clamp(EPS, 1-EPS)
    cls_prob_new = cls_prob_new.clamp(EPS, 1-EPS)

    proposals = _get_graph_centers(boxes, cls_prob, im_labels)

    labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes,
            proposals, im_labels, cls_prob_new)

    return {'labels' : labels.reshape(1, -1),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1),
            'gt_assignment' : gt_assignment.reshape(1, -1),
            'pc_labels' : pc_labels.reshape(1, -1),
            'pc_probs' : pc_probs.reshape(1, -1),
            'pc_count' : pc_count.reshape(1, -1),
            'img_cls_loss_weights' : img_cls_loss_weights.reshape(1, -1),
            'im_labels_real' : torch.cat((torch.tensor([[1.]], dtype=im_labels.dtype, device=im_labels.device), im_labels), dim=1)}


class PCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pcl_probs, labels, cls_loss_weights,
                gt_assignment, pc_labels, pc_probs, pc_count,
                img_cls_loss_weights, im_labels):
        ctx.pcl_probs = pcl_probs
        ctx.labels = labels
        ctx.cls_loss_weights = cls_loss_weights
        ctx.gt_assignment = gt_assignment
        ctx.pc_labels = pc_labels
        ctx.pc_probs = pc_probs
        ctx.pc_count = pc_count
        ctx.img_cls_loss_weights = img_cls_loss_weights
        ctx.im_labels = im_labels
        
        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_loss_weights,
                                    gt_assignment, pc_labels, pc_probs,
                                    pc_count, img_cls_loss_weights, im_labels)

        for c in im_labels.nonzero()[:,1]:
            if c == 0:
                i = (labels[0,:] == 0).nonzero()[:,0]
                loss -= (cls_loss_weights[0, i] * pcl_probs[i,c].log()).sum()
            else:
                i = (pc_labels[0,:] == c).nonzero()[:,0]
                loss -= (img_cls_loss_weights[0, i] * pc_probs[0,i].log()).sum()

        return loss / batch_size
    
    @staticmethod
    def backward(ctx, grad_output):
        pcl_probs = ctx.pcl_probs
        labels = ctx.labels
        cls_loss_weights = ctx.cls_loss_weights
        gt_assignment = ctx.gt_assignment
        pc_labels = ctx.pc_labels
        pc_probs = ctx.pc_probs
        pc_count = ctx.pc_count
        img_cls_loss_weights = ctx.img_cls_loss_weights
        im_labels = ctx.im_labels

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        batch_size, channels = pcl_probs.size()

        for c in im_labels.nonzero()[:,1]:
            i = (labels[0] == c)
            if c == 0:
                grad_input[i, c] = -cls_loss_weights[0, i] / pcl_probs[i, c]
            else:
                pc_index = gt_assignment[0, i]
                if (c != pc_labels[0, pc_index]).all():
                    print('labels mismatch.')
                grad_input[i, c] = -img_cls_loss_weights[0, pc_index] / (pc_count[0, pc_index] * pc_probs[0, pc_index])

        grad_input /= batch_size
        return grad_input, None, None, None, None, None, None, None, None


def scs(boxes):
    left_inside = boxes[:,0].view(-1,1) > boxes[:,0].view(1,-1)
    top_inside = boxes[:,1].view(-1,1) > boxes[:,1].view(1,-1)
    right_inside = boxes[:,2].view(-1,1) < boxes[:,2].view(1,-1)
    bottom_inside = boxes[:,3].view(-1,1) < boxes[:,3].view(1,-1)
    
    surrounded = left_inside & right_inside & top_inside & bottom_inside
    surrounded = surrounded.any(0)
    
    return (~surrounded).nonzero()[:,0]


def scs_label(rois, predictions, image_labels):
    rois = rois.to(predictions[0].device)
    predictions = [p[:,1:] if p.size(-1) > image_labels.size(-1) else p for p in predictions]
    predictions = torch.stack(predictions)
    gt_classes = torch.zeros((0,), dtype=torch.long, device=predictions.device)
    gt_scores = torch.zeros((0,), dtype=predictions.dtype, device=predictions.device)
    gt_boxes = torch.zeros((0,4), dtype=rois.dtype, device=rois.device)
    for c in image_labels.nonzero()[:,1]:
        top_scores, top_idxs = predictions[:,:,c].max(dim=1)
        top_boxes = rois[top_idxs.flatten(0)]
        keep = scs(top_boxes)
        
        gt_scores = torch.cat([gt_scores, top_scores[keep]])
        gt_classes = torch.cat([gt_classes, torch.full_like(keep, c+1, device=gt_classes.device)])
        gt_boxes = torch.cat([gt_boxes, top_boxes[keep]])
        
        predictions[:,top_idxs[keep],:] = 0

    keep = ops.boxes.batched_nms(gt_boxes, gt_scores, gt_classes, 0.1)
    gt_classes, gt_scores, gt_boxes = gt_classes[keep], gt_scores[keep], gt_boxes[keep]
    
    max_overlap, gt_assignment = ops.box_iou(rois, gt_boxes).max(dim=1)
    pseudo_labels = gt_classes.gather(-1, gt_assignment)
    pseudo_labels[max_overlap < 0.5] = 0
    weights = gt_scores.gather(-1, gt_assignment)
    
    return pseudo_labels, weights


def midn_loss(prediction, image_labels, reduction):
    image_prediction = prediction.sum(dim=0, keepdim=True).clamp(EPS, 1-EPS)
    return F.binary_cross_entropy(image_prediction, image_labels, reduction=reduction)


def wsddn_loss(predictions, rois, image_labels, **kwargs):
    losses = {}
    image_labels = image_labels.to(predictions[0].device)
    
    for i, prediction in enumerate(predictions):
        losses['midn' + str(i) + '_loss'] = midn_loss(prediction, image_labels, reduction='sum')

    return losses


def oicr_loss(predictions, rois, image_labels, **kwargs):
    losses = {}
    dev = predictions[0].device
    image_labels = image_labels.to(dev)
    
    losses['midn_loss'] = midn_loss(predictions[0], image_labels, reduction='mean')
    
    pseudo_labels, weights = oicr_label(rois, predictions[0], image_labels)
    i = 0
    for prediction in predictions[1:-1]:
        losses['ref' + str(i) + '_loss'] = weighted_softmax_with_loss(prediction, pseudo_labels, weights)
        pseudo_labels, weights = oicr_label(rois, prediction.softmax(-1), image_labels)
        i += 1
    losses['ref' + str(i) + '_loss'] = weighted_softmax_with_loss(predictions[-1], pseudo_labels, weights)

    return losses


def pcl_loss(predictions, rois, image_labels, **kwargs):
    losses = {}
    dev = predictions[0].device
    image_labels = image_labels.to(dev)
    
    losses['midn_loss'] = midn_loss(predictions[0], image_labels, reduction='mean')
    
    prev = predictions[0]
    pcl = PCLFunction.apply
    for i, pred in enumerate(predictions[1:]):
        pred = pred.softmax(-1)
        dct = pcl_label(rois, prev, image_labels, pred)
        args = [pred] + list(dct.values())
        losses['ref' + str(i) + '_loss'] = pcl(*args)
        prev = pred
        
    return losses


def instability_loss(predictions, rois, image_labels, **kwargs):
    losses = {}
    
    for i, prediction in enumerate(predictions[:3]):
        losses['midn' + str(i) + '_loss'] = midn_loss(prediction, image_labels, reduction='mean')

    pseudo_labels, weights = scs_label(rois, predictions[:3], image_labels)
    i = 0
    for prediction in predictions[3:-1]:
        losses['ref' + str(i) + '_loss'] = weighted_softmax_with_loss(prediction, pseudo_labels, weights)
        pseudo_labels, weights = oicr_label(rois, prediction.softmax(-1), image_labels)
        i += 1
    losses['ref' + str(i) + '_loss'] = weighted_softmax_with_loss(predictions[-1], pseudo_labels, weights)
    
    return losses


def gt_injected_midn_loss(predictions, rois, gt_boxes, gt_classes):
    max_overlap, gt_assignment = ops.box_iou(rois, gt_boxes).max(dim=1)
    fg = max_overlap >= EPS
    bg = ~fg
    loss = -(1-predictions[bg]).log().mean()
    
    for c in gt_classes.unique():
        targets = gt_assignment == c and fg
        weight = max_overlap[targets]
        weight = (normed - normed.min()) / (normed.max() - normed.min())
        loss -= (weight * predictions[targets].log()).mean()

    return loss


def gt_injected_oicr_loss(predictions, rois, gt_boxes, gt_classes):
    losses['midn_loss'] = gt_injected_midn_loss(predictions[0], rois, gt_boxes, gt_classes)
    
    for i, prediction in enumerate(predictions[1:]):
        max_overlap, gt_assignment = ops.box_iou(rois, gt_boxes).max(dim=1)
        class_assignment = gt_classes.gather(-1, gt_assignment) + 1
        class_assignment[max_overlap < 0.5] = 0
        max_overlap[max_overlap < 0.5] = 1 - max_overlap[max_overlap < 0.5]
        losses['ref' + str(i) + '_loss'] = -(max_overlap * predictions.log_softmax().gather(-1, class_assignment)).mean()

    return losses


def semi_supervised_oicr_loss(predictions, rois, image_labels, **kwargs):
    if 'gt_boxes' in kwargs and 'gt_classes' in kwargs:
        losses = gt_injected_oicr_loss(predictions, rois, kwargs['gt_boxes'], kwargs['gt_classes'])
    else:
       losses = oicr_loss(predictions, rois, image_labels, **kwargs)

    return losses


LOSS_FUNCTIONS = {
    'wsddn_loss': wsddn_loss,
    'oicr_loss': oicr_loss,
    'pcl_loss': pcl_loss,
    'instability_loss': instability_loss,
    'semi_supervised_oicr_loss': semi_supervised_oicr_loss
}
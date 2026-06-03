import math

import numpy as np


def mean_precision(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    mAP = [0] * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)
        val = n_ii / float(n_ij)
        if math.isnan(val):
            mAP[i] = 0.
        else:
            mAP[i] = val
    # print(mAP)
    return mAP


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    return IU

import torch

import numpy as np

def ece_dynamic(
    gt: np.ndarray,                # (256,256) ground truth (0=background, 1=dynamic)
    pred: np.ndarray,              # (1,2,256,256) softmax probabilities
    num_bins,
    binning_strategy: str = "equal_size",  # or "equal_population"
) -> float:
    """
    Compute Expected Calibration Error (ECE) for the DYNAMIC class (channel=1)
    using NumPy only.

    Returns
    -------
    float : scalar ECE value
    """
    assert pred.shape[0] == 2, f"pred must be (2,H,W), got {pred.shape}"
    assert gt.shape == pred.shape[-2:], f"gt must match pred spatial dims, got {gt.shape}"

    # dynamic class probabilities
    p_dyn = pred[1]  # (H, W)

    # flatten
    p = p_dyn.flatten()  # (N,)
    y = gt.flatten()     # (N,)

    # === Brier score ===
    #brier = np.mean((p - y) ** 2)
    # binary labels: 1 if dynamic else 0
    y_dyn = (y == 1).astype(np.float32)

    # choose bin boundaries
    if binning_strategy == "equal_size":
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    elif binning_strategy == "equal_population":
        sorted_p = np.sort(p)
        n = len(p)
        bin_edges = np.interp(np.linspace(0, n, num_bins + 1),
                              np.arange(n),
                              sorted_p)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        raise ValueError("binning_strategy must be 'equal_size' or 'equal_population'")

    ece = 0.0

    # compute |confidence - accuracy| weighted by bin proportion
    for i in range(num_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        in_bin = (p > lower) & (p <= upper) if i > 0 else (p >= lower) & (p <= upper)

        if np.any(in_bin):
            prop = np.mean(in_bin)
            acc  = np.mean(y_dyn[in_bin])
            conf = np.mean(p[in_bin])
            ece += abs(conf - acc) * prop

    return float(ece)

def brier_dynamic(
    gt: np.ndarray,                # (256,256) ground truth (0=background, 1=dynamic)
    pred: np.ndarray,              # (1,2,256,256) softmax probabilities
) -> float:
    """
    Compute Expected Calibration Error (ECE) for the DYNAMIC class (channel=1)
    using NumPy only.

    Returns
    -------
    float : scalar ECE value
    """
    assert pred.shape[0] == 2, f"pred must be (2,H,W), got {pred.shape}"
    assert gt.shape == pred.shape[-2:], f"gt must match pred spatial dims, got {gt.shape}"

    # dynamic class probabilities
    p_dyn = pred[1]  # (H, W)

    # flatten
    p = p_dyn.flatten()  # (N,)
    y = gt.flatten()     # (N,)

    # === Brier score ===
    brier = np.mean((p - y) ** 2)
    return brier

'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def cal_iou_training(batch_dict, output_dict):
    """
    Calculate IoU during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The iou for static and dynamic bev map.
    """

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):

        gt_static = \
            batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[i, 0]
        gt_static = np.array(gt_static, dtype=np.int)

        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic, dtype=np.int)

        pred_static = \
            output_dict['static_map'].detach().cpu().data.numpy()[i]
        pred_static = np.array(pred_static, dtype=np.int)

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic, dtype=np.int)

        iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
        iou_static = mean_IU(pred_static, gt_static)

        return iou_dynamic, iou_static


def cal_ece_brier_score(batch_dict, output_dict):
    """
    Calculate ece

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The ece for dynamic bev map.
    """

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):
        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic, dtype=np.int)

        pred_dynamic = \
            output_dict['dynamic_prob'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic, dtype=np.int)

        return ece_dynamic(gt_dynamic,pred_dynamic,num_bins=30), ece_dynamic(gt_dynamic,pred_dynamic, num_bins=30,binning_strategy='equal_population'),brier_dynamic(gt_dynamic,pred_dynamic)



def cal_nll_brier_score(batch_dict, output_dict):
    """
    Calculate nll

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The nll for dynamic bev map.
    """

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):
        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic, dtype=np.int)

        pred_dynamic = \
            output_dict['dynamic_prob'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic, dtype=np.int)

        return nll_brier(gt_dynamic,pred_dynamic)

from sklearn.metrics import brier_score_loss, log_loss

def nll_brier(gt_dynamic, pred_dynamic, ignore_index=255):
    """
    gt_dynamic:   (256, 256) with {0,1,255}
    pred_dynamic: (2, 256, 256) logits

    Returns:
        nll, brier
    """
    print(gt_dynamic.shape, pred_dynamic.shape)  # (256,256) (2,256,256)

    # ---- derive prediction + confidence from logits ----
    l0 = pred_dynamic[0]      # (256,256)
    l1 = pred_dynamic[1]      # (256,256)

    # stable softmax over 2 classes
    mx = np.maximum(l0, l1)
    exp0 = np.exp(l0 - mx)
    exp1 = np.exp(l1 - mx)
    sum_exp = exp0 + exp1

    p0 = exp0 / sum_exp       # P(class 0)
    p1 = exp1 / sum_exp       # P(class 1)

    # predicted class and its confidence
    pred = (p1 > p0).astype(np.int64)   # p_np equivalent
    conf = np.maximum(p0, p1)           # conf for predicted class (like in _process_uncertainty)

    # ---- apply valid mask, flatten (exact same pattern) ----
    t_np = gt_dynamic
    p_np = pred

    valid_mask = t_np != ignore_index
    flat_t = t_np[valid_mask].flatten()
    flat_p = p_np[valid_mask].flatten()
    flat_conf = conf[valid_mask].flatten()

    if flat_t.size == 0:
        print("No valid pixels for NLL/Brier")
        return np.nan, np.nan

    # if you want to mimic clamp_and_log_values: just clamp here
    flat_conf = np.clip(flat_conf, 1e-7, 1 - 1e-7)

    # ---- same as in _process_uncertainty ----
    acc_map = (flat_p == flat_t).astype(np.uint8)
    brier = brier_score_loss(acc_map, flat_conf)

    try:
        nll = log_loss(acc_map, flat_conf, labels=[0, 1])
    except ValueError as e:
        if "Only one class" in str(e):
            nll = np.nan
            print("Single-class NLL error in nll_brier")
        else:
            raise

    return nll, brier
    
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

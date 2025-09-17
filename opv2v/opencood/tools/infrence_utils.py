import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.tools.train_utils import save_bev_seg_binary, STD, MEAN

import argparse
import os
import statistics
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
from opencood.tools.train_utils import STD, MEAN  # used for image denormalization

# -------------------------
# Utility functions
# -------------------------
def normalize_to_uint8(img, eps=1e-6):
    """
    Normalize a numpy float image to [0,255] uint8.
    img: np.ndarray float (H,W) or (H,W,C)
    """
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < eps:
        return (np.clip(img, 0., 1.) * 255.0).astype(np.uint8)
    img_norm = (img - mn) / (mx - mn)
    return (img_norm * 255.0).astype(np.uint8)

def var_to_uncertainty_image(var_tensor, method="sqrt_sum",
                             to_numpy=True):
    """
    Convert var tensor (C,H,W) or (H,W) to a single-channel uncertainty map.
    method:
      - "sqrt_sum": sqrt(sum_c var_c)  (default)
      - "max": max_c var_c
    returns float numpy array (H,W) of uncertainties.
    """
    v = var_tensor.copy()
    if v.ndim == 3:
        # assume (C,H,W)
        if method == "sqrt_sum":
            u = np.sqrt(np.sum(v, axis=0))
        elif method == "max":
            u = np.max(v, axis=0)
        else:
            u = np.sqrt(np.sum(v, axis=0))
    elif v.ndim == 2:
        u = np.sqrt(v)
    else:
        raise ValueError("var_tensor must be 2D or 3D")
    return u

def apply_colormap_jet(gray_uint8):
    """
    Apply OpenCV JET colormap to single-channel uint8 array.
    Returns BGR uint8 image.
    """
    return cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)

def overlay_heatmap_on_image(rgb_img_uint8, heatmap_bgr_uint8, alpha=0.6):
    """
    Overlay heatmap (BGR) on rgb image (RGB->converted to BGR for cv2).
    rgb_img_uint8: H,W,3 (RGB)
    heatmap_bgr_uint8: H,W,3 (BGR)
    Returns RGB uint8 overlay.
    """
    # convert rgb->bgr
    bgr = cv2.cvtColor(rgb_img_uint8, cv2.COLOR_RGB2BGR)
    ov = ((1.0 - alpha) * bgr.astype(np.float32) +
          alpha * heatmap_bgr_uint8.astype(np.float32))
    ov = np.clip(ov, 0, 255).astype(np.uint8)
    # convert back to RGB
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

# -------------------------
# Visualization that understands uncertainty
# -------------------------
def camera_uncertainty_visualization(model_mean_var_out,
                                     post_processed_output,
                                     batch_dict,
                                     save_path,
                                     idx,
                                     model_type='dynamic',
                                     image_width=800,
                                     image_height=600,
                                     overlay_alpha=0.6):
    """
    model_mean_var_out: dict returned directly from model(...) BEFORE dataset.post_process
      expected keys (depending on target): 'dynamic_seg','dynamic_var', 'static_seg','static_var'
      shapes typically: (B,L,C,H,W) for *_seg and *_var
    post_processed_output: output_dict after dataset.post_process(...) (same as your original script uses)
      expected to contain 'dynamic_map' or 'static_map' keys (the display preds)
    batch_dict: the original batch (e.g. batch_data['ego'])
    Saves visualization to save_path/test_vis_uncert/%04d.png
    """
    output_folder = os.path.join(save_path, 'test_vis_uncert')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # raw camera inputs - same as your original
    raw_images = batch_dict['inputs'].detach().cpu().data.numpy()[0, 0]  # C,H,W per camera tile
    # We'll build a horizontal montage: all input cams + gt + pred + uncertainty + overlay
    n_cams = raw_images.shape[0]
    total_cols = n_cams + 4  # inputs + GT + PRED + UNC + OVERLAY
    canvas = np.zeros((image_height, image_width * total_cols, 3), dtype=np.uint8)

    # place raw input images
    for j in range(n_cams):
        raw_image = 255 * ((raw_images[j] * STD) + MEAN)
        raw_image = np.array(raw_image, dtype=np.uint8)
        # rgb = bgr in dataset -> convert to rgb for correct plotting
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (image_width, image_height))
        canvas[:, image_width * j:image_width * (j + 1)] = raw_image

    # choose which keys to use
    pred_key = 'dynamic_map' if model_type == 'dynamic' else 'static_map'
    gt_key = 'gt_dynamic' if model_type == 'dynamic' else 'gt_static'

    # post_processed_output contains the display prediction (e.g. binary map or color-coded)
    # We place GT and PRED at positions n_cams and n_cams+1
    # GT
    if model_type == 'dynamic':
        gt = batch_dict['gt_dynamic'].detach().cpu().data.numpy()[0, 0]  # H, W
        gt_img = np.array(gt * 255., dtype=np.uint8)
        gt_img = cv2.resize(gt_img, (image_width, image_height))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    else:
        gt_origin = batch_dict['gt_static'].detach().cpu().data.numpy()[0, 0]
        gt_img = np.zeros((gt_origin.shape[0], gt_origin.shape[1], 3), dtype=np.uint8)
        gt_img[gt_origin == 1] = np.array([88, 128, 255])
        gt_img[gt_origin == 2] = np.array([244, 148, 0])
        gt_img = cv2.resize(gt_img, (image_width, image_height))

    canvas[:, image_width * n_cams:image_width * (n_cams + 1)] = gt_img

    # PRED (post_processed_output expected to contain pred maps at full size)
    # this is what your original camera_inference_visualization used (post_process result)
    pred_disp = post_processed_output.get(pred_key, None)
    if pred_disp is not None:
        # pred_disp expected shape [B, H, W] or [B, C, H, W] depending on dataset
        pd = pred_disp.detach().cpu().data.numpy()[0]
        if pd.ndim == 3:
            # If shape is (C,H,W), convert to 2D by argmax
            pd = np.argmax(pd, axis=0)
        pd = np.array(pd * 255., dtype=np.uint8) if pd.max() <= 1.0 else pd.astype(np.uint8)
        pd = cv2.resize(pd, (image_width, image_height))
        pd = cv2.cvtColor(pd, cv2.COLOR_GRAY2BGR)
    else:
        # fallback: if post_processed didn't include predicted map, attempt to use model_mean_var_out
        pd = None

    if pd is not None:
        canvas[:, image_width * (n_cams + 1):image_width * (n_cams + 2)] = pd

    # UNCERTAINTY map: compute from model_mean_var_out
    # dynamic_var is expected as (B,L,C,H,W) -> convert to numpy and squeeze indices [0,0,...]
    if model_type == 'dynamic':
        var_t = model_mean_var_out.get('dynamic_var', None)
        mean_t = model_mean_var_out.get('dynamic_seg', None)
    else:
        var_t = model_mean_var_out.get('static_var', None)
        mean_t = model_mean_var_out.get('static_seg', None)

    if var_t is not None:
        # bring to CPU numpy
        var_np = var_t.detach().cpu().numpy()  # shape (B,L,C,H,W) or (B,L,H,W)
        # pick first sample: b=0, l=0
        if var_np.ndim == 5:
            _, _, C, H, W = var_np.shape
            var_sel = var_np[0, 0]   # (C,H,W)
        elif var_np.ndim == 4:
            # (B,L,H,W)
            var_sel = var_np[0, 0]   # (H,W)
        else:
            raise RuntimeError("Unexpected var tensor dim: {}".format(var_np.shape))

        uncert_map = var_to_uncertainty_image(var_sel, method="sqrt_sum")
        # normalize to uint8
        uncert_uint8 = normalize_to_uint8(uncert_map)
        # colorize
        uncert_color = apply_colormap_jet(uncert_uint8)
        # resize to display size
        uncert_color_resized = cv2.resize(uncert_color, (image_width, image_height))
        canvas[:, image_width * (n_cams + 2):image_width * (n_cams + 3)] = cv2.cvtColor(uncert_color_resized, cv2.COLOR_BGR2RGB)

        # overlay uncertainty onto the first raw camera view (we choose camera 0)
        rgb_raw0 = canvas[:, 0:image_width]  # already RGB
        overlay_rgb = overlay_heatmap_on_image(rgb_raw0, uncert_color_resized, alpha=overlay_alpha)
        canvas[:, image_width * (n_cams + 3):image_width * (n_cams + 4)] = overlay_rgb
    else:
        # fill empty placeholders
        blank = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        canvas[:, image_width * (n_cams + 2):image_width * (n_cams + 4)] = blank

    out_file = os.path.join(output_folder, '%04d.png' % idx)
    cv2.imwrite(out_file, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_file



def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def camera_inference_visualization(output_dict,
                                   batch_dict,
                                   output_dir,
                                   epoch,
                                   model_type='dynamic'):
    image_width = 800
    image_height = 600

    output_folder = os.path.join(output_dir, 'test_vis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_images = \
        batch_dict['ego']['inputs'].detach().cpu().data.numpy()[0, 0]
    visualize_summary = np.zeros((image_height,
                                  image_width * 6,
                                  3),
                                 dtype=np.uint8)

    for j in range(raw_images.shape[0]):
        raw_image = 255 * ((raw_images[j] * STD) + MEAN)
        raw_image = np.array(raw_image, dtype=np.uint8)
        # rgb = bgr
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (image_width, image_height))

        visualize_summary[:, image_width * j:image_width * (j + 1)] = raw_image

    if model_type == 'dynamic':
        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[0,
                                                                        0]
        gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)
        gt_dynamic = cv2.resize(gt_dynamic, (image_width,
                                             image_height))
        gt_dynamic = cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[0]
        pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)
        pred_dynamic = cv2.resize(pred_dynamic, (image_width,
                                                 image_height))
        pred_dynamic = cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)
        visualize_summary[:, image_width * 4:image_width * 5] = gt_dynamic
        visualize_summary[:, image_width * 5:] = pred_dynamic

    else:
        gt_static_origin = \
            batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[0, 0]
        gt_static = np.zeros((gt_static_origin.shape[0],
                              gt_static_origin.shape[1],
                              3), dtype=np.uint8)
        gt_static[gt_static_origin == 1] = np.array([88, 128, 255])
        gt_static[gt_static_origin == 2] = np.array([244, 148, 0])

        pred_static_origin = \
            output_dict['static_map'].detach().cpu().data.numpy()[0]
        pred_static = np.zeros((pred_static_origin.shape[0],
                                pred_static_origin.shape[1],
                                3), dtype=np.uint8)
        pred_static[pred_static_origin == 1] = np.array([88, 128, 255])
        pred_static[pred_static_origin == 2] = np.array([244, 148, 0])

        gt_static = cv2.resize(gt_static, (image_width,
                                           image_height))
        pred_static = cv2.resize(pred_static, (image_width,
                                               image_height))

        visualize_summary[:, image_width * 4:image_width * 5] = gt_static
        visualize_summary[:, image_width * 5:] = pred_static

    cv2.imwrite(os.path.join(output_folder, '%04d.png')
                % epoch, visualize_summary)

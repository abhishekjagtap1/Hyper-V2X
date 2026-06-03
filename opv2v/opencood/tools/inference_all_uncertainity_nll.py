import argparse
import statistics
import time
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import json
import sys, os
# --- remove old paths ---
bad_paths = [
    '/data/s2/abhi_workspace/CoBEVT/opv2v'
]
for p in bad_paths:
    if p in sys.path:
        sys.path.remove(p)
new_path = '/data/s2/abhi_workspace/sanath/Hyper-V2X/opv2v'
if new_path not in sys.path:
    sys.path.insert(0, new_path)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training , cal_ece_brier_score,cal_nll_brier_score

from torchmetrics.functional.classification import binary_calibration_error
#python -m pip install torchmetrics
# =====================
# Visualization helpers
# =====================

print(sys.path)


def normalize_to_uint8(img):
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)

def apply_colormap_jet(img_uint8):
    return cv2.applyColorMap(img_uint8, cv2.COLORMAP_TURBO)

def overlay_heatmap_on_image(rgb_image, heatmap_color, alpha=0.6):
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def var_to_uncertainty_image(var_map):
    """
    Converts variance tensor (C,H,W) or (H,W) into a scalar uncertainty map.
    For multi-channel (e.g. epistemic), we sum across channels then take sqrt.
    """
    if var_map.ndim == 3:
        var_scalar = np.sqrt(var_map.sum(axis=0))
    else:
        var_scalar = np.sqrt(var_map)
    return var_scalar


import os
import cv2
import math
import numpy as np
import torch

import os
import cv2
import math
import numpy as np
import torch

import os
import cv2
import math
import numpy as np
import torch

# ---- fixed uncertainty scales (from dataset-level stats) ----
EPI_VMIN   = 0.0
EPI_VMAX   = 0.23      # from epi_max mean ≈ 0.223...
ALEO_VMIN  = 0.0
ALEO_VMAX  = 0.70      # from aleo_max mean ≈ 0.69...
TOTAL_VMIN = 0.0
TOTAL_VMAX = 0.92      # from tot_max mean ≈ 0.91...


def camera_uncertainty_visualization_final(model_mean_var_out,
                                           post_processed_output,
                                           batch_dict,
                                           save_path,
                                           idx,
                                           model_type='dynamic',
                                           image_width=800,
                                           image_height=600,
                                           overlay_alpha=0.6):
    """
    Uses batch_dict['raw_inputs'][0] (dict: agent_id -> (4,H,W,3)).

    Saves (in save_path/Hyper_V2X_Compression_2/):
      - RAW per-camera images (ORIGINAL RES, NO LABELS): {idx:04d}_agent{AGENT}_cam{K}.png
      - GT (NO LABEL):          {idx:04d}_gt.png
      - Prediction (NO LABEL):  {idx:04d}_pred.png
      - Epistemic (NO LABEL):   {idx:04d}_unc_epistemic.png
      - Aleatoric (NO LABEL):   {idx:04d}_unc_aleatoric.png
      - Total (NO LABEL):       {idx:04d}_unc_total.png
      - Composite (labels on all tiles): {idx:04d}_composite.png

    Now uses FIXED vmin/vmax for uncertainty maps so different models
    are visually comparable.
    """
    out_dir = os.path.join(save_path, 'Hyper_V2X_Compression_2')
    os.makedirs(out_dir, exist_ok=True)

    # -------- RAW inputs dict --------
    raw_inputs_dict = batch_dict['raw_inputs'][0]  # {agent_id: (4,H,W,3)}
    agent_ids = sorted(raw_inputs_dict.keys())

    # Flatten: [(agent_id, cam_idx, img(H,W,3)), ...]
    flat_samples = []
    for agent_id in agent_ids:
        imgs4 = raw_inputs_dict[agent_id]
        if torch.is_tensor(imgs4):
            imgs4 = imgs4.detach().cpu().numpy()
        assert imgs4.ndim == 4 and imgs4.shape[-1] == 3, \
            f"Got {imgs4.shape} for agent {agent_id}, expected (4,H,W,3)"
        for cam_idx in range(imgs4.shape[0]):
            flat_samples.append((agent_id, cam_idx, imgs4[cam_idx]))

    n_inputs = len(flat_samples)

    # -------- Helpers --------
    def to_uint8_rgb_minmax(img):
        """(H,W,3) float/uint8 -> uint8 RGB [0,255] via min-max if float."""
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr
        m, M = float(arr.min()), float(arr.max())
        if M - m < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        arr = (arr - m) / (M - m)
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def put_label(tile_bgr, text):
        """Overlay label on a BGR tile already resized to (image_width, image_height)."""
        overlay = tile_bgr.copy()
        cv2.rectangle(overlay, (0, 0),
                      (int(0.6 * image_width), 36),
                      (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.35, tile_bgr, 0.65, 0)
        cv2.putText(out, text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def tensor_to_hwc_uint8_gray(x, vmin=None, vmax=None):
        """
        Torch/np -> uint8 gray HxW.

        If vmin/vmax given, use FIXED scaling (for consistency across models).
        If not, fall back to per-map min/max (legacy behavior).
        """
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        while x.ndim > 2:
            x = x[0]

        if vmin is None:
            vmin = float(x.min())
        if vmax is None:
            vmax = float(x.max())

        if vmax - vmin < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)

        x = np.clip(x, vmin, vmax)
        return ((x - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

    def tensor_to_colorjet(x, vmin=None, vmax=None):
        """Scalar map -> JET BGR uint8."""
        gray = tensor_to_hwc_uint8_gray(x, vmin=vmin, vmax=vmax)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # BGR

    # -------- 1) Save per-image RAW (NO LABELS, ORIGINAL RES) + build inputs grid --------
    INPUTS_COLS = min(8, max(1, n_inputs))
    inputs_rows = math.ceil(n_inputs / INPUTS_COLS)
    inputs_block_h = inputs_rows * image_height
    inputs_block_w = INPUTS_COLS * image_width
    inputs_block = np.zeros((inputs_block_h, inputs_block_w, 3), dtype=np.uint8)

    for idx_in, (agent_id, cam_idx, img_orig) in enumerate(flat_samples):
        img_uint8_rgb = to_uint8_rgb_minmax(img_orig)
        img_uint8_bgr = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR)
        raw_name = os.path.join(
            out_dir, f"{idx:04d}_agent{agent_id}_cam{cam_idx}.png"
        )
        cv2.imwrite(raw_name, img_uint8_bgr)

        r = idx_in // INPUTS_COLS
        c = idx_in % INPUTS_COLS
        tile_bgr = cv2.resize(img_uint8_bgr, (image_width, image_height))
        tile_bgr = put_label(tile_bgr, f"agent {agent_id} | cam {cam_idx}")
        inputs_block[
            r * image_height:(r + 1) * image_height,
            c * image_width:(c + 1) * image_width
        ] = tile_bgr

    # -------- 2) Build GT + prediction maps --------
    if model_type == 'dynamic':
        gt = batch_dict['gt_dynamic']
        gt_np = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
        gt_np = gt_np[0, 0]  # [B,1,H,W] -> [H,W]
        gt_u8 = (gt_np * 255.0).astype(np.uint8)
        gt_u8 = cv2.resize(gt_u8, (image_width, image_height))
        gt_bgr = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
    else:
        gt = batch_dict['gt_static']
        gt_np = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
        gt_origin = gt_np[0, 0]
        gt_bgr = np.zeros((gt_origin.shape[0], gt_origin.shape[1], 3),
                          dtype=np.uint8)
        gt_bgr[gt_origin == 1] = (255, 128, 88)  # BGR
        gt_bgr[gt_origin == 2] = (0, 148, 244)
        gt_bgr = cv2.resize(gt_bgr, (image_width, image_height))
    cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_gt.png"), gt_bgr)

    pred_key = 'dynamic_map' if model_type == 'dynamic' else 'static'
    pred_disp = post_processed_output.get(pred_key, None)
    if pred_disp is not None:
        pd = pred_disp.detach().cpu().numpy() if torch.is_tensor(pred_disp) else np.asarray(pred_disp)
        pd = pd[0]  # first sample
        if pd.ndim == 3:
            pd = np.argmax(pd, axis=0)
        pd_u8 = (pd * 255.0).astype(np.uint8) if pd.max() <= 1.0 else pd.astype(np.uint8)
        pd_u8 = cv2.resize(pd_u8, (image_width, image_height))
        pd_bgr = cv2.cvtColor(pd_u8, cv2.COLOR_GRAY2BGR)
    else:
        pd_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_pred.png"), pd_bgr)

    # -------- 3) Uncertainties with FIXED scales --------
    var_t   = model_mean_var_out.get('dynamic_var' if model_type == 'dynamic' else 'static_var', None)
    aleo_t  = model_mean_var_out.get('dynamic_aleo' if model_type == 'dynamic' else 'static_aleo', None)
    total_t = model_mean_var_out.get('total_unc'  if model_type == 'dynamic' else 'total_unc_static', None)

    if var_t is not None:
        epi_bgr = tensor_to_colorjet(var_t,
                                     vmin=EPI_VMIN,
                                     vmax=EPI_VMAX)
        epi_bgr = cv2.resize(epi_bgr, (image_width, image_height))
    else:
        epi_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_unc_epistemic.png"), epi_bgr)

    if aleo_t is not None:
        aleo_bgr = tensor_to_colorjet(aleo_t,
                                      vmin=ALEO_VMIN,
                                      vmax=ALEO_VMAX)
        aleo_bgr = cv2.resize(aleo_bgr, (image_width, image_height))
    else:
        aleo_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_unc_aleatoric.png"), aleo_bgr)

    if total_t is not None:
        total_bgr = tensor_to_colorjet(total_t,
                                       vmin=TOTAL_VMIN,
                                       vmax=TOTAL_VMAX)
        total_bgr = cv2.resize(total_bgr, (image_width, image_height))
    else:
        total_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_unc_total.png"), total_bgr)

    # -------- 4) Compose final canvas with labels --------
    MAPS_COLS = 5
    maps_block_h = image_height
    maps_block_w = MAPS_COLS * image_width
    maps_block = np.zeros((maps_block_h, maps_block_w, 3), dtype=np.uint8)

    def add_label(tile_bgr, label):
        return put_label(tile_bgr, label)

    maps_block[:, 0*image_width:1*image_width] = add_label(gt_bgr.copy(),   "GT")
    maps_block[:, 1*image_width:2*image_width] = add_label(pd_bgr.copy(),   "Prediction")
    maps_block[:, 2*image_width:3*image_width] = add_label(epi_bgr.copy(),  "Epistemic")
    maps_block[:, 3*image_width:4*image_width] = add_label(aleo_bgr.copy(), "Aleatoric")
    maps_block[:, 4*image_width:5*image_width] = add_label(total_bgr.copy(),"Total")

    canvas_h = inputs_block_h + maps_block_h
    canvas_w = max(inputs_block_w, maps_block_w)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    x_inputs = (canvas_w - inputs_block_w) // 2
    x_maps   = (canvas_w - maps_block_w) // 2

    canvas[0:inputs_block_h,
           x_inputs:x_inputs + inputs_block_w] = inputs_block
    canvas[inputs_block_h:inputs_block_h + maps_block_h,
           x_maps:x_maps + maps_block_w] = maps_block

    out_file = os.path.join(out_dir, f"{idx:04d}_composite.png")
    cv2.imwrite(out_file, canvas)
    return out_file

# ============================
# Visualization main function
# ============================
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
    Visualize input cameras + GT + Pred + Epistemic + Aleatoric + Total uncertainty.
    """
    output_folder = os.path.join(save_path, 'Hyper_V2X_Compression_2')
    os.makedirs(output_folder, exist_ok=True)

    raw_images = batch_dict['inputs'].detach().cpu().data.numpy()[0, 0]
    n_cams = raw_images.shape[0]

    total_cols = n_cams + 5  # Inputs + GT + PRED + 3 uncertainty maps
    canvas = np.zeros((image_height, image_width * total_cols, 3), dtype=np.uint8)

    # --- 1. RAW CAMERAS ---
    MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
    STD = np.array([0.229, 0.224, 0.225])[None, None, :]

    for j in range(n_cams):
        raw_image = 255 * ((raw_images[j] * STD) + MEAN)
        raw_image = np.array(raw_image, dtype=np.uint8)
        # rgb = bgr in dataset -> convert to rgb for correct plotting
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (image_width, image_height))
        canvas[:, image_width * j:image_width * (j + 1)] = raw_image

    # --- 2. GROUND TRUTH ---
    if model_type == 'dynamic':
        gt = batch_dict['gt_dynamic'].detach().cpu().data.numpy()[0, 0]
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

    # --- 3. PREDICTION ---
    pred_key = 'dynamic_map' if model_type == 'dynamic' else 'static'
    pred_disp = post_processed_output.get(pred_key, None)
    if pred_disp is not None:
        pd = pred_disp.detach().cpu().data.numpy()[0]
        if pd.ndim == 3:
            pd = np.argmax(pd, axis=0)
        pd = np.array(pd * 255., dtype=np.uint8) if pd.max() <= 1.0 else pd.astype(np.uint8)
        pd = cv2.resize(pd, (image_width, image_height))
        pd = cv2.cvtColor(pd, cv2.COLOR_GRAY2BGR)
        canvas[:, image_width * (n_cams + 1):image_width * (n_cams + 2)] = pd

    # --- 4. UNCERTAINTIES ---
    # Epistemic
    var_t = model_mean_var_out.get('dynamic_var' if model_type == 'dynamic' else 'static_var', None)
    # Aleatoric
    aleo_t = model_mean_var_out.get('dynamic_aleo' if model_type == 'dynamic' else 'static_aleo', None)
    # Total
    total_t = model_mean_var_out.get('total_unc' if model_type == 'dynamic' else 'total_unc_static', None)

    def _unc_to_color(unc_t):
        if unc_t is None:
            return np.zeros((image_height, image_width, 3), dtype=np.uint8)
        unc_np = unc_t.detach().cpu().numpy()
        if unc_np.ndim == 5:
            unc_np = unc_np[0, 0]
        elif unc_np.ndim == 4:
            unc_np = unc_np[0, 0]
        unc_map = var_to_uncertainty_image(unc_np)
        unc_uint8 = normalize_to_uint8(unc_map)
        unc_color = apply_colormap_jet(unc_uint8)
        unc_color = cv2.resize(unc_color, (image_width, image_height))
        unc_color = cv2.cvtColor(unc_color, cv2.COLOR_BGR2RGB)
        return unc_color

    epistemic_color = _unc_to_color(var_t)
    aleatoric_color = _unc_to_color(aleo_t)
    total_color = _unc_to_color(total_t)

    # Place epistemic
    canvas[:, image_width * (n_cams + 2):image_width * (n_cams + 3)] = epistemic_color
    # Place aleatoric
    canvas[:, image_width * (n_cams + 3):image_width * (n_cams + 4)] = aleatoric_color
    # Place total
    canvas[:, image_width * (n_cams + 4):image_width * (n_cams + 5)] = total_color

    out_file = os.path.join(output_folder, f'{idx:04d}.png')
    cv2.imwrite(out_file, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_file

IGNORE_INDEX = 255  # void label to ignore

def brier_per_image(prob_map, gt_map):
    """
    prob_map: (B,H,W) float in [0,1]  (foreground probability)
    gt_map:   (B,H,W) long in {0,1,255}
    returns:  list[float] length B, per-image Brier score
    """
    assert prob_map.shape == gt_map.shape
    B = prob_map.shape[0]
    scores = []
    for b in range(B):
        y  = gt_map[b]
        p  = prob_map[b]
        m  = (y != IGNORE_INDEX)
        if m.any():
            yb = (y[m] == 1).float()
            pb = p[m]
            scores.append(float(((pb - yb) ** 2).mean()))
        else:
            scores.append(0.0)
    return scores



def summarize_uncertainty_stats():
    print("\n=== Dataset-Level Uncertainty Stats (means over samples) ===")
    for key, values in uncertainty_dataset_stats.items():
        if len(values) == 0:
            print(f"{key:10s}: no data")
            continue
        print(f"{key:10s}: mean={np.mean(values):.6f}, "
              f"min={np.min(values):.6f}, max={np.max(values):.6f}")

# ============================
# Inference Loop
# ============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--config', required=False)
    parser.add_argument('--model_type', type=str, default='dynamic', choices=['dynamic','static'])
    parser.add_argument('--out_dir', type=str, default='inference_out/Demo_IV_checkpoint')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_width', type=int, default=800)
    parser.add_argument('--image_height', type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    fake_opt = argparse.Namespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, fake_opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()




    static_ave_iou = []
    dynamic_ave_iou = []
    lane_ave_iou = []
    ece_dynamic_list = []
    ece_dynamic_eqp_list=[]
    brier_score_list= []
    brier_score_listv2= []
    nll_list= []
    os.makedirs(args.out_dir, exist_ok=True)

    for i, batch_data in enumerate(data_loader):
        #if i > 1022:
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                print("agent_ids order:", batch_data['ego']['agent_ids'])
                print("ego_id:", batch_data['ego']['ego_id'])
                # camera_data[0] corresponds to ego
                model_out = model(batch_data['ego'])
                post_output = opencood_dataset.post_process(batch_data['ego'], model_out)

                # vis_file = camera_uncertainty_visualization_final(
                #     model_out,
                #     post_output,
                #     batch_data['ego'],
                #     os.path.join(args.model_dir,args.out_dir),
                #     i,
                #     model_type=args.model_type,
                #     image_width=args.image_width,
                #     image_height=args.image_height
                # )
                # print(f'Saved visualization to {vis_file}')

                iou_dynamic, iou_static = cal_iou_training(batch_data, post_output)
                print('iou_dynamic for',i ,':',iou_dynamic)
                ece_dynamic, ece_dynamic_eqp, brier_score_redundant = cal_ece_brier_score(batch_data, post_output)
                nll, brier=cal_nll_brier_score(batch_data, post_output)
                static_ave_iou.append(iou_static[1])
                dynamic_ave_iou.append(iou_dynamic[1])
                lane_ave_iou.append(iou_static[2])
                ece_dynamic_list.append(ece_dynamic)
                ece_dynamic_eqp_list.append(ece_dynamic_eqp)
                brier_score_list.append(brier_score_redundant)
                brier_score_listv2.append(brier)
                nll_list.append(nll)

    static_ave_iou = statistics.mean(static_ave_iou) if static_ave_iou else 0.0
    dynamic_ave_iou = statistics.mean(dynamic_ave_iou) if dynamic_ave_iou else 0.0
    lane_ave_iou = statistics.mean(lane_ave_iou) if lane_ave_iou else 0.0

    #mean_ece_static = statistics.mean(ece_static_list) if ece_static_list else 0.0
    mean_ece_dynamic = statistics.mean(ece_dynamic_list) if ece_dynamic_list else 0.0
    mean_ece_dynamic_eqp = statistics.mean(ece_dynamic_eqp_list) if ece_dynamic_eqp_list else 0.0
    # mean_brier_static  = statistics.mean(brier_static_list)  if brier_static_list  else 0.0
    mean_brier_dynamic = statistics.mean(brier_score_list) if brier_score_list else 0.0
    mean_brier_dynamicv2 = statistics.mean(brier_score_listv2) if brier_score_listv2 else 0.0
    mean_nll = statistics.mean(nll_list) if nll_list else 0.0


    print('Road IoU: %f' % static_ave_iou)
    print('Lane IoU: %f' % lane_ave_iou)
    print('Dynamic IoU: %f' % dynamic_ave_iou)

    #print(f"Static Mean ECE:   {mean_ece_static:.4f}")
    print(f"Dynamic Mean ECE:  {mean_ece_dynamic:.4f}")
    print(f"Dynamic ECE Equal Population:  {mean_ece_dynamic_eqp:.4f}")
    #print(f"Static Mean Brier: {mean_brier_static:.4f}")
    print(f"Dynamic nll :{mean_nll:.4f}")
    print(f"Dynamic Mean Brier Score :{mean_brier_dynamicv2:.4f}")



if __name__ == '__main__':
    main()

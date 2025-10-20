import argparse
import statistics
import time
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training

# =====================
# Visualization helpers
# =====================

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


# ============================
# Inference Loop
# ============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--config', required=False)
    parser.add_argument('--model_type', type=str, default='dynamic', choices=['dynamic','static'])
    parser.add_argument('--out_dir', type=str, default='inference_out/GAP_K8')
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

    os.makedirs(args.out_dir, exist_ok=True)

    for i, batch_data in enumerate(data_loader):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            model_out = model(batch_data['ego'])
            post_output = opencood_dataset.post_process(batch_data['ego'], model_out)

            vis_file = camera_uncertainty_visualization(
                model_out,
                post_output,
                batch_data['ego'],
                args.out_dir,
                i,
                model_type=args.model_type,
                image_width=args.image_width,
                image_height=args.image_height
            )
            print(f'Saved visualization to {vis_file}')

            iou_dynamic, iou_static = cal_iou_training(batch_data, post_output)
            static_ave_iou.append(iou_static[1])
            dynamic_ave_iou.append(iou_dynamic[1])
            lane_ave_iou.append(iou_static[2])

    static_ave_iou = statistics.mean(static_ave_iou) if static_ave_iou else 0.0
    dynamic_ave_iou = statistics.mean(dynamic_ave_iou) if dynamic_ave_iou else 0.0
    lane_ave_iou = statistics.mean(lane_ave_iou) if lane_ave_iou else 0.0

    print('Road IoU: %f' % static_ave_iou)
    print('Lane IoU: %f' % lane_ave_iou)
    print('Dynamic IoU: %f' % dynamic_ave_iou)
    print(f'Saved visualizations to {args.out_dir}/test_vis_uncert')


if __name__ == '__main__':
    main()

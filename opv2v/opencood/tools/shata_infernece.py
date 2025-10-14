import argparse
import statistics
import time
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
import opencood.tools.infrence_utils as infrence_utils

import cv2
import numpy as np
import os

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def visualize_uncertainty_map(pred_uncertainty, base_map, image_width, image_height,
                              thresh_ratio=0.3, alpha=0.5):
    """
    pred_uncertainty: np.array [H, W]
    base_map: np.array [H, W] or [H, W, 3]
    """
    # 1. Normalize uncertainty
    max_val = pred_uncertainty.max()
    if max_val > 0:
        pred_uncertainty = pred_uncertainty / max_val
    pred_uncertainty = np.clip(pred_uncertainty, 0, 1)

    # 2. Optional: threshold to emphasize uncertain regions only
    mask_high = (pred_uncertainty >= thresh_ratio).astype(np.uint8)

    # 3. Convert to heatmap (custom colormap)
    heatmap = (pred_uncertainty * 255).astype(np.uint8)
    # Instead of COLORMAP_JET, use HOT or TURBO for sharper gradient
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # 4. Make uncertainty edges crisp
    edges = cv2.Canny((pred_uncertainty * 255).astype(np.uint8), 100, 200)
    heatmap[edges > 0] = [255, 255, 255]  # draw edges in white

    # 5. Overlay with base map (segmentation)
    if base_map.ndim == 2:
        base_map = np.uint8(base_map * 127)
        base_map = cv2.cvtColor(base_map, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(base_map, 1 - alpha, heatmap, alpha, 0)

    # 6. Optional: dim out low-uncertainty regions
    overlay[mask_high == 0] = (0.7 * overlay[mask_high == 0]).astype(np.uint8)

    # 7. Resize once for display
    overlay_resized = cv2.resize(overlay, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return overlay_resized

def camera_inference_visualization_uncertainty(output_dict,
                                               batch_dict,
                                               output_dir,
                                               epoch,
                                               model_type='dynamic'):
    image_width = 800
    image_height = 600

    output_folder = os.path.join(output_dir, 'test_vis_compression0_pakka_64')
    os.makedirs(output_folder, exist_ok=True)

    # ---------------- RAW CAMERA INPUT PANELS ----------------
    raw_images = batch_dict['ego']['inputs'].detach().cpu().numpy()[0, 0]
    visualize_summary = np.zeros((image_height,
                                  image_width * 7,
                                  3),
                                 dtype=np.uint8)

    for j in range(raw_images.shape[0]):
        raw_image = 255 * ((raw_images[j] * STD) + MEAN)
        raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (image_width, image_height))
        visualize_summary[:, image_width * j:image_width * (j + 1)] = raw_image

    # ==========================================================
    # ---------------------- DYNAMIC MAP ------------------------
    # ==========================================================
    if model_type == 'dynamic':
        # Ground truth dynamic map
        gt_dynamic = batch_dict['ego']['gt_dynamic'][0, 0].cpu().numpy()
        gt_dynamic = np.uint8(gt_dynamic * 255)
        gt_dynamic = cv2.resize(gt_dynamic, (image_width, image_height))
        gt_dynamic = cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)

        # Prediction dynamic map
        pred_dynamic = output_dict['dynamic_map'][0].cpu().numpy()
        pred_dynamic = np.uint8(pred_dynamic * 255)
        pred_dynamic = cv2.resize(pred_dynamic, (image_width, image_height))
        pred_dynamic = cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)

        visualize_summary[:, image_width * 4:image_width * 5] = gt_dynamic
        visualize_summary[:, image_width * 5:image_width * 6] = pred_dynamic

        # Uncertainty
        pred_uncertainty = output_dict.get('dynamic_var', None)
        if pred_uncertainty is not None:
            pred_uncertainty = pred_uncertainty[0, 0].cpu().numpy()
            pred_uncertainty = pred_uncertainty.mean(axis=0)

            # Match resolution with dynamic_map
            H_var, W_var = pred_uncertainty.shape
            H_map, W_map = output_dict['dynamic_map'].shape[1:]
            if (H_var, W_var) != (H_map, W_map):
                pred_uncertainty = cv2.resize(pred_uncertainty, (W_map, H_map), interpolation=cv2.INTER_LINEAR)

            # Visualization
            overlay_resized = visualize_uncertainty_map(
                pred_uncertainty,
                output_dict['dynamic_map'][0].cpu().numpy(),
                image_width,
                image_height,
                thresh_ratio=0.35,  # 🔥 highlight only top uncertainty
                alpha=0.6
            )
            visualize_summary[:, image_width * 6:] = overlay_resized

    # ==========================================================
    # ----------------------- STATIC MAP -----------------------
    # ==========================================================
    else:
        # Ground truth static map
        gt_static_origin = batch_dict['ego']['gt_static'][0, 0].cpu().numpy()
        gt_static = np.zeros((*gt_static_origin.shape, 3), dtype=np.uint8)
        gt_static[gt_static_origin == 1] = [88, 128, 255]
        gt_static[gt_static_origin == 2] = [244, 148, 0]

        # Prediction static map
        pred_static_origin = output_dict['static_map'][0].cpu().numpy()
        pred_static = np.zeros((*pred_static_origin.shape, 3), dtype=np.uint8)
        pred_static[pred_static_origin == 1] = [88, 128, 255]
        pred_static[pred_static_origin == 2] = [244, 148, 0]

        gt_static = cv2.resize(gt_static, (image_width, image_height))
        pred_static = cv2.resize(pred_static, (image_width, image_height))

        visualize_summary[:, image_width * 4:image_width * 5] = gt_static
        visualize_summary[:, image_width * 5:image_width * 6] = pred_static

        # Uncertainty
        pred_uncertainty = output_dict.get('static_var', None)
        if pred_uncertainty is not None:
            pred_uncertainty = pred_uncertainty[0, 0].cpu().numpy()

            if pred_uncertainty.ndim == 3:
                pred_uncertainty = pred_uncertainty.mean(axis=0)
            elif pred_uncertainty.ndim == 0 or pred_uncertainty.size == 0:
                pred_uncertainty = np.zeros_like(output_dict['static_map'][0].cpu().numpy())

            H, W = output_dict['static_map'].shape[1], output_dict['static_map'].shape[2]
            pred_uncertainty = cv2.resize(pred_uncertainty, (W, H))

            max_val = pred_uncertainty.max()
            if max_val > 0:
                pred_uncertainty = pred_uncertainty / max_val
            pred_uncertainty = np.uint8(pred_uncertainty * 255)

            pred_uncertainty = cv2.applyColorMap(pred_uncertainty, cv2.COLORMAP_JET)
            pred_uncertainty = cv2.resize(pred_uncertainty, (image_width, image_height))
            visualize_summary[:, image_width * 6:] = pred_uncertainty

    # ==========================================================
    # ----------------------- SAVE IMAGE -----------------------
    # ==========================================================
    save_path = os.path.join(output_folder, f'{epoch:04d}.png')
    cv2.imwrite(save_path, visualize_summary)



def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--model_type', type=str, default='dynamic',
                        help='dynamic or static prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of MC samples for uncertainty')
    return parser.parse_args()


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    dynamic_ave_iou = []
    static_ave_iou = []
    lane_ave_iou = []

    for i, batch_data in enumerate(data_loader):
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            # 🟡 Inference with uncertainty (K controls # of weight samples)
            output_dict = model(batch_data['ego'])  # K is handled inside the model

            # Post-processing
            output_dict = opencood_dataset.post_process(batch_data['ego'], output_dict)

            # Visualization with uncertainty
            camera_inference_visualization_uncertainty(
                output_dict, batch_data, opt.model_dir, i, opt.model_type
            )

            iou_dynamic, iou_static = cal_iou_training(batch_data, output_dict)
            static_ave_iou.append(iou_static[1])
            dynamic_ave_iou.append(iou_dynamic[1])
            lane_ave_iou.append(iou_static[2])

    print('Road IoU:', statistics.mean(static_ave_iou))
    print('Lane IoU:', statistics.mean(lane_ave_iou))
    print('Dynamic IoU:', statistics.mean(dynamic_ave_iou))


if __name__ == '__main__':
    main()

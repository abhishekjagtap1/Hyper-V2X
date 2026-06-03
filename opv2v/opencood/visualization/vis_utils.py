import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

from opencood.utils import box_utils
from opencood.utils import common_utils
import os
import math
import torch

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs

def linset_assign_list(vis,
                       lineset_list1,
                       lineset_list2,
                       update_mode='update'):
    """
    Associate two lists of lineset.

    Parameters
    ----------
    vis : open3d.Visualizer
    lineset_list1 : list
    lineset_list2 : list
    update_mode : str
        Add or update the geometry.
    """
    for j in range(len(lineset_list1)):
        index = j if j < len(lineset_list2) else -1
        lineset_list1[j] = \
            lineset_assign(lineset_list1[j],
                                     lineset_list2[index])
        if update_mode == 'add':
            vis.add_geometry(lineset_list1[j])
        else:
            vis.update_geometry(lineset_list1[j])


def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """

    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1


def color_encoding(intensity, mode='intensity'):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == 'constant':
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color


def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      save_path='',
                                      mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1.0

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)


def visualize_sequence_sample_output(pred_tensor_list,
                                     gt_tensor_list,
                                     pcd_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()

    while True:
        for i, (pred_tensor, gt_tensor, pcd) in \
                enumerate(zip(pred_tensor_list, gt_tensor_list, pcd_list)):
            pred_tensor = pred_tensor.copy()
            gt_tensor = gt_tensor.copy()
            pcd = pcd.copy()

            pcd_intcolor = color_encoding(pcd[:, -1])
            pcd[:, :1] = -pcd[:, :1]
            vis_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
            vis_pcd.colors = o3d.utility.Vector3dVector(pcd_intcolor)

            oabbs_pred = bbx2oabb(pred_tensor, 'hwl')
            oabbs_gt = bbx2oabb(gt_tensor, 'hwl', color=(0, 1, 0))
            oabbs = oabbs_pred + oabbs_gt

            if i == 0:
                vis.add_geometry(vis_pcd)

            for oabb in oabbs:
                vis.add_geometry(oabb)

            vis.update_geometry(vis_pcd)

            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('pinhole_param.json')
            ctr.convert_from_pinhole_camera_parameters(param)

            vis.poll_events()
            vis.update_renderer()

            for oabb in oabbs:
                vis.remove_geometry(oabb)
            time.sleep(0.01)
    vis.destroy_window()


def visualize_single_sample_output_bev(pred_box, gt_box, pcd, dataset,
                                       show_vis=True,
                                       save_path=''):
    """
    Visualize the prediction, groundtruth with point cloud together in
    a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor
        (N, 4, 2) prediction.

    gt_box : torch.Tensor
        (N, 4, 2) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.
    """

    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = \
        np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = (((bbx - bev_origin)) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(batch_data,
                                       o3d_pcd,
                                       order,
                                       key='origin_lidar',
                                       visualize=False,
                                       save_path='',
                                       oabb=False,
                                       mode='constant'):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    o3d_pcd : o3d.PointCloud
        Open3d PointCloud.

    order : str
        The bounding box order.

    key : str
        origin_lidar for late fusion and stacked_lidar for early fusion.
        todo: consider intermediate fusion in the future.

    visualize : bool
        Whether to visualize the sample.

    batch_data : dict
        The dictionary that contains current timestamp's data.

    save_path : str
        If set, save the visualization image to the path.

    oabb : bool
        If oriented bounding box is used.
    """

    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
        bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)

    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

    return o3d_pcd, aabbs


def visualize_inference_sample_dataloader(pred_box_tensor,
                                          gt_box_tensor,
                                          origin_lidar,
                                          o3d_pcd,
                                          mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """

    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader, order, color_mode='constant'):
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.Dataloader
        Pytorch dataloader

    order : str
        Bounding box order(N, 7).

    color_mode : str
        Color rendering mode.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = \
                visualize_single_sample_dataloader(sample_batched['ego'],
                                                   vis_pcd,
                                                   order,
                                                   mode=color_mode)
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.add_geometry(vis_aabbs[i])

            for i in range(len(vis_aabbs)):
                index = i if i < len(aabbs) else -1
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                vis.update_geometry(vis_aabbs[i])

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    vis.destroy_window()


def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def visualize_bev(batch_data):
    bev_input = batch_data["processed_lidar"]["bev_input"]
    label_map = batch_data["label_dict"]["label_map"]
    if not isinstance(bev_input, np.ndarray):
        bev_input = common_utils.torch_tensor_to_numpy(bev_input)

    if not isinstance(label_map, np.ndarray):
        label_map = label_map[0].numpy() if not label_map[0].is_cuda else \
            label_map[0].cpu().detach().numpy()

    if len(bev_input.shape) > 3:
        bev_input = bev_input[0, ...]

    plt.matshow(np.sum(bev_input, axis=0))
    plt.axis("off")
    plt.matshow(label_map[0, :, :])
    plt.axis("off")
    plt.show()



# ---- fixed uncertainty scales (from dataset-level stats) ----
EPI_VMIN   = 0.0
EPI_VMAX   = 0.23      # from epi_max mean ≈ 0.223...
ALEO_VMIN  = 0.0
ALEO_VMAX  = 0.70      # from aleo_max mean ≈ 0.69...
TOTAL_VMIN = 0.0
TOTAL_VMAX = 0.92      # from tot_max mean ≈ 0.91...


def summarize_uncertainty_stats():
    print("\n=== Dataset-Level Uncertainty Stats (means over samples) ===")
    for key, values in uncertainty_dataset_stats.items():
        if len(values) == 0:
            print(f"{key:10s}: no data")
            continue
        print(f"{key:10s}: mean={np.mean(values):.6f}, "
              f"min={np.min(values):.6f}, max={np.max(values):.6f}")


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



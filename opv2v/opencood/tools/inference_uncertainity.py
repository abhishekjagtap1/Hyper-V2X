import argparse
import statistics
import time

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
import os
from opencood.tools.infrence_utils import camera_uncertainty_visualization





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='path where checkpoint is stored (same as your train saved path)')
    parser.add_argument('--config', required=False, help='(optional) path to yaml config; if not provided the loader from train_utils will use model_dir metadata')
    parser.add_argument('--model_type', type=str, default='dynamic', choices=['dynamic','static'], help='which head to visualize')
    parser.add_argument('--out_dir', type=str, default='inference_out', help='where to save visualizations')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_width', type=int, default=800)
    parser.add_argument('--image_height', type=int, default=600)
    return parser.parse_args()

def main():
    args = parse_args()

    # build hyper-parameters & dataset
    # replicate how your training script loads yaml
    dummy = argparse.Namespace()  # used by yaml_utils.load_yaml - your code used hypes = yaml_utils.load_yaml(None, opt)
    # Provide model_dir in a manner compatible with yaml_utils
    # We'll call yaml_utils.load_yaml with None and a fake namespace that has model_dir attribute if required by your codebase
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
    saved_path = args.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    dynamic_ave_iou = []
    static_ave_iou = []
    lane_ave_iou = []

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for i, batch_data in enumerate(data_loader):
        print('Processing idx', i)
        with torch.no_grad():
            # prepare & move to device
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_data = train_utils.to_device(batch_data, device)

            # 1) Run model and capture raw mean/var outputs (before post_process)
            model_out = model(batch_data['ego'])   # this should be the dict from CorpBEVT.forward -> HyperBevSegHead outputs

            # 2) Save or use the raw mean/var maps for uncertainty visualization
            # NOTE: model_out keys likely: dynamic_seg, dynamic_var, static_seg, static_var
            # We pass model_out directly into our visualization routine below.

            # 3) Post-process as before for display & IoU
            post_output = opencood_dataset.post_process(batch_data['ego'], model_out)

            # 4) Visualize: overlay predictions and uncertainty
            vis_file = camera_uncertainty_visualization(model_out,
                                                       post_output,
                                                       batch_data['ego'],
                                                       out_dir,
                                                       i,
                                                       model_type=args.model_type,
                                                       image_width=args.image_width,
                                                       image_height=args.image_height,
                                                       overlay_alpha=0.6)
            print('Saved visualization to', vis_file)

            # 5) Compute IoU metrics exactly as before (uses post_processed maps)
            iou_dynamic, iou_static = cal_iou_training(batch_data, post_output)
            static_ave_iou.append(iou_static[1])
            dynamic_ave_iou.append(iou_dynamic[1])
            # lane class index may exist in static calculation (same as your script)
            lane_ave_iou.append(iou_static[2])

    static_ave_iou = statistics.mean(static_ave_iou) if len(static_ave_iou) > 0 else 0.0
    dynamic_ave_iou = statistics.mean(dynamic_ave_iou) if len(dynamic_ave_iou) > 0 else 0.0
    lane_ave_iou = statistics.mean(lane_ave_iou) if len(lane_ave_iou) > 0 else 0.0

    print('Road IoU: %f' % static_ave_iou)
    print('Lane IoU: %f' % lane_ave_iou)
    print('Dynamic IoU: %f' % dynamic_ave_iou)
    print('Saved visualizations to %s/test_vis_uncert' % out_dir)

if __name__ == '__main__':
    main()
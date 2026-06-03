import argparse
import os
import sys
import numpy as np
import torch

# -----------------------------
# REMOVE BAD PATHS
# -----------------------------
bad_paths = ['/data/s2/abhi_workspace/CoBEVT/opv2v']
for p in bad_paths:
    if p in sys.path:
        sys.path.remove(p)

new_path = '/data/s2/abhi_workspace/PhD_IV_2026_abhi/Hyper-V2X/opv2v'
if new_path not in sys.path:
    sys.path.insert(0, new_path)


from torch.utils.data import DataLoader
from opencood.visualization.vis_utils import camera_uncertainty_visualization

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import (
    cal_iou_training,
    cal_ece_brier_score,
    cal_nll_brier_score
)


# -----------------------------
# RICH IMPORTS
# -----------------------------
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

# -----------------------------
# RUNNING MEAN
# -----------------------------
class RunningMean:
    def __init__(self):
        self.n = 0
        self.val = 0.0

    def update(self, x):
        self.n += 1
        self.val += (x - self.val) / self.n

    def get(self):
        return self.val


# -----------------------------
# ARGS
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='inference_out')
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Save visualization outputs'
    )
    return parser.parse_args()


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    # ---------------- LOAD CONFIG ----------------
    fake_opt = argparse.Namespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, fake_opt)

    print("Building dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch,
        pin_memory=False,
        drop_last=False
    )

    print("Loading model...")
    model = train_utils.create_model(hypes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)

    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    # ---------------- METRICS ----------------
    iou_dyn_m = RunningMean()
    iou_lane_m = RunningMean()
    nll_m = RunningMean()
    brier_m = RunningMean()
    ece_m = RunningMean()

    total = len(loader)

    # ---------------- RICH PROGRESS ----------------
    with torch.no_grad():
        with Live(refresh_per_second=4, console=console) as live:
            with Progress(
                TextColumn("Inference"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:

                task = progress.add_task("run", total=total)

                for i, batch_data in enumerate(loader):

                    batch_data = train_utils.to_device(batch_data, device)

                    model_out = model(batch_data['ego'])
                    post_output = dataset.post_process(batch_data['ego'], model_out)

                    if args.save_vis:
                        vis_file = camera_uncertainty_visualization(
                            model_out,
                            post_output,
                            batch_data['ego'],
                            os.path.join(args.model_dir,args.out_dir),
                            i,
                            model_type='dynamic',
                            image_width=800,
                            image_height=600,
                        )

                    # Segmentation Metric
                    iou_d, iou_s = cal_iou_training(batch_data, post_output)

                    ##### Uncertainty Metrics #########
                    ece, ece_eqp, _ = cal_ece_brier_score(batch_data, post_output)
                    nll, brier = cal_nll_brier_score(batch_data, post_output)

                    iou_dyn_m.update(iou_d[1])
                    iou_lane_m.update(iou_s[2])
                    nll_m.update(nll)
                    brier_m.update(brier)
                    ece_m.update(ece)

                    # update progress
                    progress.update(task, advance=1)

                    # build live table
                    table = Table(title="Running Metrics")

                    table.add_column("Metric")
                    table.add_column("Value")

                    table.add_row("IoU (Dynamic)", f"{iou_dyn_m.get():.4f}")
                    table.add_row("IoU (Lane)", f"{iou_lane_m.get():.4f}")
                    table.add_row("NLL", f"{nll_m.get():.4f}")
                    table.add_row("Brier", f"{brier_m.get():.4f}")
                    table.add_row("ECE", f"{ece_m.get():.4f}")

                    live.update(table)


    # ---------------- FINAL RESULTS ----------------
    print("\n================ FINAL RESULTS ================\n")
    print(f"Dynamic IoU : {iou_dyn_m.get():.4f}")
    print(f"Lane IoU    : {iou_lane_m.get():.4f}")
    print(f"NLL         : {nll_m.get():.4f}")
    print(f"Brier       : {brier_m.get():.4f}")
    print(f"ECE         : {ece_m.get():.4f}")


if __name__ == "__main__":
    main()

import os
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from VoxelDataset import VoxelDataset
from P2VNet import P2VModel, loss_p2v, JointTrainigModel

VOXEL_SIZE = 0.3
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)  # Ensure workers use fork (default on Linux), otherwise each worker would preload separately.

# Limit OpenMP threads per worker to avoid CPU over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)


# ---------- Hyperparameters ----------
@dataclass
class TrainConfig:
    # data folder (output of the training items extracted from the ply file)
    root_dir: str = "/home/larry/codeGit/p2v-slam/Python/Output/training_items"

    # model key settings.
    voxel_size: float = VOXEL_SIZE
    origin_jitter_ratio: float = 0.0    # voxel_size=0.4 -> ±0.1m. Default: 
    origin_jitter_prob: float = 0.0     # Above this probability, no jitter. Else, add random (VOXEL_SIZE*ratio) jitter.
    loss_type: str = 'vec'              # vec, dir_dist
    nll_on: bool = True                 # 
    save_dir: str = "/home/larry/codeGit/p2v-slam/Python/Output/model_output"
    save_name: str = f"joint-model.pth"
    
    # model other settings
    u_pe_dim: int = 21
    feat_dim: int = 64
    point_noise_std: float = 0.01
    
    # Training settings
    num_workers: int = 16               # How many workers for training. 
    prefetch_factor: int = 4
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    decay_step: int = 50
    decay: float = 0.50
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Not used settings.
    n_aug: int = 1                      # Currently base-only: only generates label_0, so must be 1
    weight_z: bool = False              # Add weight on z-axis

    # tensor board
    log_root: str = "/home/larry/codeGit/p2v-net/log"
    log_comment: str = f"log"


cfg = TrainConfig()




@torch.no_grad()
def eval_one_epoch(model: P2VModel, loader, device):
    model.eval()
    loss_tot_ave = loss_1_ave = loss_2_ave = 0.0

    flag_once = False
    for batch in loader:
        points_27 = batch["points_27"].to(device, non_blocking=True)
        exist_mask = batch["exist_mask_27"].to(device, non_blocking=True)
        query = batch["query"].to(device, non_blocking=True)
        gt_p2v = batch["p2v"].to(device, non_blocking=True)
        # delta_27 = batch["deltas_27"].to(device, non_blocking=True)
        points_mask_27 = batch["points_mask_27"].to(device, non_blocking=True)

        # u_pe = fourier_pe_u(query)
        pred_p2v, sigma_d, _, _ = model(points_27, query, exist_mask, points_mask_27)

        loss_tot, loss_1, loss_2 = loss_p2v(pred_p2v, gt_p2v, sigma_d, mode=cfg.loss_type, use_nll=cfg.nll_on, weight_z=cfg.weight_z)

        loss_tot_ave += loss_tot.item()
        loss_1_ave += loss_1.item()
        loss_2_ave += loss_2.item()
        
        if not flag_once:     # Output one data for evaluation.
            one_gt = gt_p2v[0].detach().cpu().numpy()
            one_pred = pred_p2v[0].detach().cpu().numpy()
            one_sigma = sigma_d[0].detach().cpu().numpy()
            print(f"    [Eval sample] gt:[{one_gt[0]:.4f},{one_gt[1]:.4f},{one_gt[2]:.4f}] "
                  f"pred:[{one_pred[0]:.4f},{one_pred[1]:.4f},{one_pred[2]:.4f}] "
                  f"sigma_d:{one_sigma[0]:.4f}")
            flag_once = True
            
    N = max(1, len(loader))
    return loss_tot_ave / N, loss_1_ave / N, loss_2_ave / N


def train_one_epoch(model, loader, optimizer, device, scaler, cfg: TrainConfig):
    model.train()
    loss_tot_ave = loss_1_ave = loss_2_ave = 0.0
    for batch in loader:
        points_27 = batch["points_27"].to(device, non_blocking=True)
        exist_mask = batch["exist_mask_27"].to(device, non_blocking=True)
        query = batch["query"].to(device, non_blocking=True)
        gt_p2v = batch["p2v"].to(device, non_blocking=True)
        # delta_27 = batch["deltas_27"].to(device, non_blocking=True)
        points_mask_27 = batch["points_mask_27"].to(device, non_blocking=True)

        # Optional: Add noise only to valid points
        if cfg.point_noise_std > 0:
            noise = torch.randn_like(points_27) * cfg.point_noise_std
            points_27 = points_27 + noise * points_mask_27.unsqueeze(-1)

        # u_pe = fourier_pe_u(query)

        optimizer.zero_grad(set_to_none=True)

        # AMP mixed precision
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_p2v, sigma_d, _, _ = model(points_27, query, exist_mask, points_mask_27)
            # loss_tot, loss_vec, loss_dist = loss_p2v(pred_p2v, gt_p2v, sigma_d)
            loss_tot, loss_1, loss_2 = loss_p2v(pred_p2v, gt_p2v, sigma_d, mode=cfg.loss_type, use_nll=cfg.nll_on, weight_z=cfg.weight_z)

        scaler.scale(loss_tot).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_tot_ave += loss_tot.item()
        loss_1_ave += loss_1.item()
        loss_2_ave += loss_2.item()

    N = max(1, len(loader))
    return loss_tot_ave / N, loss_1_ave / N, loss_2_ave / N



def main():
    device = torch.device(cfg.device)

    print("--------------- Train_v2_online ---------------")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("----------------------------------------------")

    train_dataset = VoxelDataset(
        root_dir=cfg.root_dir,
        voxel_size=cfg.voxel_size,
        n_sample=22,
        seed=42,
        is_train=True,
        is_debug=False,
        n_aug=cfg.n_aug,
        origin_jitter_ratio=cfg.origin_jitter_ratio,
        origin_jitter_prob=cfg.origin_jitter_prob,
        enable_local_rotation=True,
    )
    test_dataset = VoxelDataset(
        root_dir=cfg.root_dir,
        voxel_size=cfg.voxel_size,
        n_sample=22,
        seed=43,
        is_train=False,   # Evaluation: no jitter
        is_debug=False,
        n_aug=cfg.n_aug,
        origin_jitter_ratio=0.0,
        origin_jitter_prob=0.0,
        enable_local_rotation=False,
    )

    # DataLoader: Keep CPU pipeline keeping up with GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=False,
    )
    
    print("--> Preloading train/test dataset.")
    train_dataset._preload_all_voxels()
    test_dataset._preload_all_voxels()
    print("<-- Done")
    

    # model = P2VModel(feat_dim=cfg.feat_dim, u_dim=cfg.u_pe_dim).to(device)
    model = JointTrainigModel(feat_dim=cfg.feat_dim, u_dim=cfg.u_pe_dim).to(device)
    # model_joint = JointTrainigModel(feat_dim=cfg.feat_dim, u_dim=cfg.u_pe_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.decay_step, gamma=cfg.decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Logger
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.log_root, f"{time_str}-{cfg.log_comment}")
    writer = SummaryWriter(log_dir)

    best_test = float("inf")
    save_path = os.path.join(cfg.save_dir, cfg.save_name)
    os.makedirs(cfg.save_dir, exist_ok=True)


    ## PhD plot save.
    log_file = "/home/larry/codeGit/p2v-net/src/AugmentationNew/train_log.txt"
    # Write header before training (only once)
    with open(log_file, "w") as f:
        f.write("epoch,train_tot,train_vec,train_nll,test_tot,test_vec,test_nll\n")
    ## PhD plot save.
    
    for epoch in range(cfg.epochs):

        t0 = time.time()

        # P2VModel
        tr_loss, tr1, tr2 = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg)
        t1 = time.time()
        te_loss, te1, te2 = eval_one_epoch(model, test_loader, device)
        t2 = time.time()
        scheduler.step()
        
        
        
        # Keep original multiplication by VOXEL_SIZE (convert back to meters)
        if cfg.loss_type == 'vec':
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}], vec loss: train={tr_loss:.4f} | test={te_loss:.4f}")
        elif cfg.loss_type == 'dir_dist':                   # Losses are: total, dist, direction.
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}] "
                f"train={tr_loss:.4f} dist={tr1:.4f} dir={tr2:.4f} | "
                f"test={te_loss:.4f} dist={te1:.4f} dir={te2:.4f}")
            
        print(f"    [Timer]: train={t1-t0:.2f}s test={t2-t1:.2f}s")

        writer.add_scalar("train/total", tr_loss, epoch)
        writer.add_scalar("test/total", te_loss, epoch)
        
        if cfg.loss_type == 'vec':
            if cfg.nll_on:
                comment1 = 'vector'
                comment2 = 'nll_d'
            else:
                comment1 = 'vector'
                comment2 = 'vector'

        # elif cfg.loss_type == 'dir_dist':
        #     comment1 = 'distance'
        #     comment2 = 'angle'
            
        
        writer.add_scalar(f"train/loss 1/{comment1}", tr1, epoch)
        writer.add_scalar(f"test/loss 1/{comment1}", te1, epoch)
        writer.add_scalar(f"train/loss 2/{comment2}", tr2, epoch)
        writer.add_scalar(f"test/loss 2/{comment2}", te2, epoch)
        
        # writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if te_loss < best_test:
            best_test = te_loss
            state = {
                "epoch": epoch,
                "cfg": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, save_path)
            print(f"    save best to {save_path}")
        
        
        
        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},"               # Epoch starts from 1
                f"{tr_loss:.6f},"
                f"{tr1:.6f},"
                f"{tr2:.6f},"
                f"{te_loss:.6f},"
                f"{te1:.6f},"
                f"{te2:.6f}\n"
            )
        
    print("done.")


if __name__ == "__main__":
    main()
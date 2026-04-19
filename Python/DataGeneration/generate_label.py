# Generate training items (and labels) from a ply file

import math
import sys
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from VoxelGridIndexer import VoxelGridIndexer
import time

VOXEL_SIZE = 0.3

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_path", type=str, default="/home/larry/codeGit/p2v-net/data/ply/room_static.ply", help="High-precision pointcloud ply input")
    ap.add_argument("--out_dir", type=str, default="/home/larry/codeGit/p2v-slam/Python/Output/training_items", help="Training items (with labels) output")
    ap.add_argument("--voxel_size", type=float, default=VOXEL_SIZE)
    ap.add_argument("--min_points_per_voxel", type=int, default=22) 
    ap.add_argument("--sample_per_voxel", type=int, default=5)      # How many samples to generate for each voxel? Increase this to generate more training data.
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--knn_max_dist", type=float, default=VOXEL_SIZE)
    ap.add_argument("--sigma_t", type=float, default=0.05)          # Jitter on tangential
    ap.add_argument("--sigma_n", type=float, default=0.05)          # Jitter on normal
    ap.add_argument("--far_ratio", type=float, default=0.1)         # How many points are set to be 3-sigma sampling
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.1)        # Separate train/test dataset
    ap.add_argument("--export_sample_ply", type=int, default=1, help="Save pointcloud of the sampled point")
    ap.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    return ap.parse_args()


args = parse_args()

# -----------------------------
# IO utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_ply_points(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts

def write_voxel_csvs(buckets: dict, out_voxel_dir: str):
    """
    buckets: {hash -> np.ndarray[N,3] world points}
    output: out_voxel_dir/<hash>.csv
    """
    ensure_dir(out_voxel_dir)
    t0 = time.time()
    
    for h, pts in tqdm(buckets.items(), desc='Writing'):
        Min_Point_For_Labeling = 22
        if pts.shape[0] < Min_Point_For_Labeling:
            continue
        out_path = os.path.join(out_voxel_dir, f"{int(h)}.csv")
        np.savetxt(out_path, pts.astype(np.float32), delimiter=",", fmt="%.6f")


# -----------------------------
# KDTree
# -----------------------------
class GlobalKNN:
    def __init__(self, points_world, leafsize=32):
        self.points = points_world.astype(np.float64)
        self.tree = KDTree(self.points, leafsize=leafsize)

    def query_knn(self, q, k):
        d, idx = self.tree.query(q[None, :], k=k)
        return self.points[idx[0], :], d[0]


# -----------------------------
# Robust quad fitting
# -----------------------------
def robust_weights_huber(res, delta):
    a = np.abs(res)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = (delta / (a[mask] + 1e-12))
    return w

def fit_quad_and_label(q_world, knn_pts, gaussian_sigma=0.3, huber_delta=0.05):
    P = knn_pts
    nbhd = P.shape[0]
    mu = P.mean(axis=0)
    X = P - mu
    C = (X.T @ X) / max(nbhd - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)
    e1 = eigvecs[:, order[2]]
    e2 = eigvecs[:, order[1]]
    e3 = eigvecs[:, order[0]]
    if np.linalg.det(np.stack([e1, e2, e3], axis=1)) < 0:
        e2 = -e2

    w_q = np.dot(q_world - mu, e3)
    q_proj = q_world - w_q * e3

    U = np.stack([np.dot(p - q_proj, e1) for p in P])
    V = np.stack([np.dot(p - q_proj, e2) for p in P])
    W = np.stack([np.dot(p - q_proj, e3) for p in P])

    R2 = U**2 + V**2 + (W**2)
    w_gauss = np.exp(-R2 / (gaussian_sigma**2 + 1e-9))

    A = np.stack([U**2, V**2, U*V, U, V, np.ones_like(U)], axis=1)

    W0 = w_gauss
    Aw = A * W0[:, None]
    zw = W * W0
    try:
        coef, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
    except np.linalg.LinAlgError:
        coef = np.zeros(6, dtype=np.float64)

    pred = A @ coef
    res = W - pred
    w_huber = robust_weights_huber(res, delta=huber_delta)
    W1 = W0 * w_huber

    Aw = A * W1[:, None]
    zw = W * W1
    try:
        coef, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
    except np.linalg.LinAlgError:
        pass

    a, b, c, d_lin, e_lin, f0 = coef.tolist()
    dz_du = d_lin
    dz_dv = e_lin
    grad_local = np.array([-dz_du, -dz_dv, 1.0], dtype=np.float64)
    grad_norm = np.linalg.norm(grad_local) + 1e-12
    n_local = grad_local / grad_norm

    basis = np.stack([e1, e2, e3], axis=1)
    n_world = (basis @ n_local).astype(np.float64)
    n_world = n_world / (np.linalg.norm(n_world) + 1e-12)

    w_q = float(np.dot(q_world - q_proj, e3))
    d_star = (w_q - f0) / grad_norm
    if d_star < 0:
        d_star = -d_star
        n_world = -n_world

    pred = A @ coef
    res = W - pred
    rmse = math.sqrt(float(np.sum(W1 * (res**2)) / (np.sum(W1) + 1e-12)))
    sigma = rmse

    r_world = d_star * n_world
    plot = {"coef": coef.astype(np.float64),
            "q_proj": q_proj.astype(np.float64),
            "e1": e1.astype(np.float64),
            "e2": e2.astype(np.float64),
            "e3": e3.astype(np.float64)}
    return (float(d_star), n_world.astype(np.float64), r_world.astype(np.float64), float(sigma), int(nbhd), plot)


# -----------------------------
# Sampling around the pointcloud surface, with jitter.
# -----------------------------
def sample_queries_near_surface_by_voxels(points_world: np.ndarray,
                                         indexer: VoxelGridIndexer,
                                         knn: GlobalKNN,
                                         sample_per_voxel: int,
                                         min_points_per_voxel: int,
                                         sigma_t: float, sigma_n: float,
                                         far_ratio: float,
                                         knn_for_pca: int,
                                         seed: int):
    """
    1) Use indexer.voxel_index to bucket the entire point cloud (origin/voxel_size consistent with training)
    2) Filter out voxels with points < min_points_per_voxel
    3) Sample K0 = ceil(sample_number / N_valid_vox) base points per voxel (take all if insufficient)
    4) Apply PCA to base points, add noise along tangent plane/normal direction to get query
    """
    rng = np.random.default_rng(seed)
    N = points_world.shape[0]

    # 1) Bucketing: hash -> indices
    buckets_idx = defaultdict(list)
    buckets_cnt = defaultdict(int)

    # For speed: compute ijk + hash for each point
    for i in tqdm(range(N), desc='1-> bucketing'):
        idx = indexer.voxel_index(points_world[i])
        h = indexer.voxel_hash(idx)
        buckets_idx[h].append(i)
        buckets_cnt[h] += 1

    # 2) Filter valid voxels
    valid_hashes = [h for h, c in buckets_cnt.items() if c >= min_points_per_voxel]
    N_valid = len(valid_hashes)
    if N_valid == 0:
        raise RuntimeError("No voxel meets min_points_per_voxel requirement, cannot generate labels.")

    K0 = sample_per_voxel   # Each voxel gets 5 points to generate labels. TODO:

    # 3) Select base points per voxel
    base_indices = []
    for h in tqdm(valid_hashes, desc='2-> random generation'):
        inds = np.asarray(buckets_idx[h], dtype=np.int64)
        cnt = inds.size
        k = min(K0, cnt)
        pick = rng.choice(inds, size=k, replace=False)
        base_indices.append(pick)

    base_indices = np.concatenate(base_indices, axis=0)
    print(f"[Sample] valid_vox={N_valid}, K0={K0}, base_points={base_indices.size}")

    # 4) Generate query points from base points (close to surface)
    Q = np.zeros((base_indices.size, 3), dtype=np.float64)
    for i, ib in enumerate(tqdm(base_indices, desc="3-> fitting")):
        x = points_world[ib].astype(np.float64)

        nbhd_pts, _ = knn.query_knn(x, k=min(knn_for_pca, max(10, N)))
        mu = nbhd_pts.mean(axis=0)
        X = nbhd_pts - mu
        C = (X.T @ X) / max(nbhd_pts.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)
        t1 = eigvecs[:, order[2]]
        t2 = eigvecs[:, order[1]]
        n  = eigvecs[:, order[0]]

        if np.linalg.norm(t1) < 1e-9 or np.linalg.norm(t2) < 1e-9 or np.linalg.norm(n) < 1e-9:
            t1 = np.array([1.0,0,0]); t2 = np.array([0,1.0,0]); n = np.array([0,0,1.0])

        a1 = rng.normal(0.0, sigma_t)
        a2 = rng.normal(0.0, sigma_t)
        sn = sigma_n if rng.random() > far_ratio else (sigma_n * 3.0)
        b  = rng.normal(0.0, sn)

        Q[i] = x + a1 * t1 + a2 * t2 + b * n

    return Q



# -----------------------------
# main
# -----------------------------

def main():
    out_dir = args.out_dir
    label_dir = out_dir
    vox_dir = os.path.join(out_dir, "voxels", "0")
    ensure_dir(label_dir)
    ensure_dir(os.path.join(out_dir, "voxels"))
    ensure_dir(vox_dir)

    print(f"Output dir: {args.out_dir}")

    print(f"[1/5] Reading original map: {args.ply_path}")
    t0 = time.time()
    pts_world = read_ply_points(args.ply_path)
    if pts_world.shape[0] == 0:
        raise RuntimeError("Empty ply.")
    print(f"  points: {pts_world.shape}")
    t_read = time.time() - t0
    print(f"[Timer] load map: {t_read}")
    
    
    indexer = VoxelGridIndexer(voxel_size=float(args.voxel_size), origin=tuple(args.origin))
    print("[2/5] Building KDTree")
    t0 = time.time()
    knn = GlobalKNN(pts_world)
    t_kdtree = time.time() - t0
    print(f"[Timer] build kdtree: {t_kdtree}")
    

    print("[3/5] Voxelizing.")
    t0 = time.time()
    # buckets: hash -> pts_world
    buckets = defaultdict(list)
    for p in tqdm(pts_world, desc='Indexing hash'):
        idx = indexer.voxel_index(p)
        h = indexer.voxel_hash(idx)
        buckets[h].append(p)
    print("[3/5] Writing to csv")
    buckets_arr = {}
    for h, plist in buckets.items():
        buckets_arr[int(h)] = np.asarray(plist, dtype=np.float32)
    write_voxel_csvs(buckets_arr, vox_dir)
    print(f"  wrote voxel files: {len(buckets_arr)}")
    t_writecsv = time.time() - t0
    print(f"[Timer] write to csv: {t_writecsv}")

    
    print("[4/4] Generating sample points.")
    t0 = time.time()
    Q = sample_queries_near_surface_by_voxels(
        points_world=pts_world,
        indexer=indexer,
        knn=knn,
        sample_per_voxel=int(args.sample_per_voxel),
        min_points_per_voxel=int(args.min_points_per_voxel),
        sigma_t=float(args.sigma_t),
        sigma_n=float(args.sigma_n),
        far_ratio=float(args.far_ratio),
        knn_for_pca=20,
        seed=int(args.seed),
    )
    t_sampling = time.time() - t0
    print(f"[Timer] sampling: {t_sampling}")
    
    if args.export_sample_ply == 1:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(Q.astype(np.float64))
        o3d.io.write_point_cloud(os.path.join(out_dir, "sample_query_base.ply"), pc)
        print("  exported sample ply.")


    print("[5/5] Generating labels and saving.")
    t0 = time.time()
    # Precompute: hash -> count, to reduce lookups
    voxel_counts = {h: buckets_arr[h].shape[0] for h in buckets_arr.keys()}
    rows = []
    cnt_no_hash = 0
    cnt_less_point = 0
    cnt_large_knn = 0
    t0 = time.time()
    for q in tqdm(Q, desc="labeling"):
        idx = indexer.voxel_index(q)
        h = int(indexer.voxel_hash(idx))

        if h not in buckets_arr:
            cnt_no_hash += 1
            continue
        if voxel_counts[h] < int(args.min_points_per_voxel):
            cnt_less_point += 1
            continue

        neighbor_pts, dists = knn.query_knn(q, k=int(args.knn))
        if float(np.mean(dists)) > float(args.knn_max_dist):
            cnt_large_knn += 1
            continue

        d_star, n_star, r_vec, sigma, _, plot_coeff = fit_quad_and_label(q, neighbor_pts, gaussian_sigma=0.3, huber_delta=0.05)

        #######################################################
        # Debug Data
        coeff = plot_coeff['coef']
        q_proj = plot_coeff['q_proj']
        e1 = plot_coeff['e1']
        e2 = plot_coeff['e2']
        e3 = plot_coeff['e3']

        center = indexer.voxel_center(idx)
        u_local = (q - center) / args.voxel_size
        rows.append({
            "voxel_hash": int(h),
            "anchor_i": int(idx[0]), "anchor_j": int(idx[1]), "anchor_k": int(idx[2]),
            "u_local_x": float(u_local[0]), "u_local_y": float(u_local[1]), "u_local_z": float(u_local[2]),
            "d_star": d_star,
            "x": float(q[0]), "y": float(q[1]), "z": float(q[2]),
            "n_x": float(n_star[0]), "n_y": float(n_star[1]), "n_z": float(n_star[2]),
            "p2v_x": float(-r_vec[0]), "p2v_y": float(-r_vec[1]), "p2v_z": float(-r_vec[2]),
            "sigma": sigma,
            "coeff_0": coeff[0], "coeff_1": coeff[1], "coeff_2": coeff[2],
            "coeff_3": coeff[3], "coeff_4": coeff[4], "coeff_5": coeff[5],
            "q_proj_0": q_proj[0], "q_proj_1": q_proj[1], "q_proj_2": q_proj[2],
            "e1_0": e1[0], "e1_1": e1[1], "e1_2": e1[2],
            "e2_0": e2[0], "e2_1": e2[1], "e2_2": e2[2],
            "e3_0": e3[0], "e3_1": e3[1], "e3_2": e3[2],
        })
        #######################################################
            

    label_col = ["voxel_hash", "anchor_i", "anchor_j", "anchor_k", "x", "y", "z", "p2v_x", "p2v_y", "p2v_z", "sigma"]
    df_label = pd.DataFrame(rows, columns=label_col)
    
    debug_col = ["voxel_hash",
                "coeff_0","coeff_1","coeff_2","coeff_3","coeff_4","coeff_5",
                "q_proj_0","q_proj_1","q_proj_2",
                "e1_0","e1_1","e1_2",
                "e2_0","e2_1","e2_2",
                "e3_0","e3_1","e3_2"]
    df_debug = pd.DataFrame(rows, columns=debug_col)
    df_debug.to_csv(os.path.join(label_dir, f"debug_0_plot.csv"), index=False)


    # Fixed to aug_idx=0
    df_label.to_csv(os.path.join(label_dir, "label_0.csv"), index=False)

    test_df  = df_label.sample(frac=float(args.test_ratio), random_state=42)
    train_df = df_label.drop(test_df.index).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df.to_csv(os.path.join(label_dir, "label_0_train.csv"), index=False)
    test_df.to_csv(os.path.join(label_dir, "label_0_test.csv"), index=False)
    
    t_label = time.time() - t0
    print(f"[Timer] label: {t_label}")


    print(f"[DONE] saved labels={len(df_label)}. skipped: no-hash={cnt_no_hash}, less-point={cnt_less_point}, too-large={cnt_large_knn}")
    print(f"  voxels: {vox_dir}")
    print(f"  labels: {label_dir}/label_0_train.csv, label_0_test.csv")
    
    print(f"---------------- Total time ----------------")
    t_total = t_read + t_kdtree + t_writecsv + t_sampling + t_label
    print(f"Load: {t_read:.2f}s, KDTree: {t_kdtree:.2f}s, WriteCSV: {t_writecsv:.2f}s, Sampling: {t_sampling:.2f}s, Label: {t_label:.2f}s, ")
    print(f"Total : {t_total:.2f}s")


if __name__ == "__main__":
    main()
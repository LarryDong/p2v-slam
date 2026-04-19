import os
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from VoxelGridIndexer import VoxelGridIndexer

import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
import MyUtils as my_utils




class VoxelDataset(Dataset):
    """
    Read label.csv and voxels/<hash>.csv, return:
      - query: [3] (world coordinates)
      - p2v: [3]
      - voxel_points: [K,3] (anchor voxel points, local normalized coordinates, unchanged)
      - neighbor_voxel_points: [26, K, 3] (neighboring 26 voxel points, local coordinates, each relative to its own voxel center)
    Optional:
      - return masks: voxel_mask [K] (True=valid point), neighbor_mask [26] (True=neighbor exists)
    Parameters
    ----
    root_dir: Path like ".../data/label" (with label.csv and voxels/ underneath)
    points_per_voxel: K, unified points per voxel (default 64)
    seed: Sampling random seed (for random downsampling when exceeding K)
    return_masks: Whether to return masks (default False)
    """
    
    def __init__(self,
                 root_dir: str,
                 voxel_size: int = 0.4,
                 n_sample: int = 64,
                 seed: int = 42,
                 is_train: bool = True,
                 is_debug: bool = False,
                 n_aug: int = 1,
                 origin_jitter_ratio: float = 0.25,     # Voxel should jitter about 25% of the voxel_size
                 origin_jitter_prob: float = 0.75,      # How many voxels are jittered
                 enable_local_rotation: bool = False):  # Apply rotation when training
        super().__init__()
        
        self.origin_jitter_ratio = float(origin_jitter_ratio)
        self.origin_jitter_prob = float(origin_jitter_prob)
        self.enable_local_rotation = enable_local_rotation
        
        self.time_acc = 0
        self.batch_cnt = 0
        self.root = root_dir
        self.n_aug = n_aug
        self.voxel_root = os.path.join(root_dir, "voxels")
        self.is_train = is_train
        self.is_debug = is_debug
        self.n_sample = n_sample
        self.n_aug = n_aug
        self.voxel_size = voxel_size
        
        
        print("------------------------------------------------------")
        print("VoxelDataset settings: ")
        print(f"  voxel_size: {self.voxel_size}")
        print(f"  n_sample: {self.n_sample}")
        print(f"  is_train: {self.is_train}")
        print(f"  jitter prob :   {self.origin_jitter_prob}")
        print(f"  jitter scale:   {self.origin_jitter_ratio}")
        print("------------------------------------------------------")
        
        if not os.path.isdir(self.voxel_root):
            raise NotADirectoryError(f"voxels root dir not found: {self.voxel_root}")

        self.indexer = VoxelGridIndexer(voxel_size=self.voxel_size)
        self.rng = np.random.default_rng(seed)

        # Small cache: hash -> np.ndarray[N,3]. Quick load
        self._cache_world: Dict[int, np.ndarray] = {}                 # Cache points for faster load
        self._cache_norm: Dict[int, np.ndarray] = {}                 # Cache points for faster load
        
        self.center_id = self.indexer.offset_to_id(0,0,0)       # Center voxel's id. Surrounding has 26.
        
        # One-hot embedding, 27 offsets
        self.delta_ids = np.array([self.indexer.offset_to_id(dx,dy,dz) for (dx,dy,dz) in self.indexer.OFFSET27])     
        self.deltas_27 = np.asarray(self.indexer.OFFSET27, dtype=np.int64)
        self.deltas_27 = torch.from_numpy(self.deltas_27)
        
        
        # Used to summarize valid sample information from all augmentations
        self.aug_idx_list: List[int] = []
        self.query_list: List[List[float]] = []
        self.p2v_list: List[List[float]] = []
        self.anchor_list: List[List[int]] = []

        total_rows_all = 0
        total_empty_all = 0

        # Iterate through each augmentation
        for aug_idx in range(self.n_aug):
            if is_train:
                label_csv = os.path.join(root_dir, f"label_{aug_idx}_train.csv")
            else:
                label_csv = os.path.join(root_dir, f"label_{aug_idx}_test.csv")
                
            if not os.path.isfile(label_csv):
                raise FileNotFoundError(f"label csv not found: {label_csv}")

            df = pd.read_csv(label_csv)
            rows = len(df)
            total_rows_all += rows
            print(f"--> [aug {aug_idx}] Loaded {rows} rows from {label_csv}")

            num_valid, num_empty_center = self._build_valid_indices_for_aug(aug_idx, df)
            total_empty_all += num_empty_center
            print(f"    [aug {aug_idx}] center-empty: {num_empty_center}(should be 0) / {rows}, valid: {num_valid}")

        # Aggregate into numpy arrays for direct index access in __getitem__
        self.aug_idx_arr = np.asarray(self.aug_idx_list, dtype=np.int64)
        self.query_arr = np.asarray(self.query_list, dtype=np.float32)
        self.p2v_arr = np.asarray(self.p2v_list, dtype=np.float32)
        self.anchor_arr = np.asarray(self.anchor_list, dtype=np.int64)

        self.num_samples = self.query_arr.shape[0]
        
        # Maintain compatibility with old code: still have valid_indices
        self.valid_indices: List[int] = list(range(self.num_samples))

        print("------------------------------------------------------")
        mode = "train" if self.is_train else "test"
        print(f"Dataset for: {mode}")
        print(f"Total rows        : {total_rows_all}")
        print(f"Center-empty total: {total_empty_all}")
        print(f"Valid samples     : {self.num_samples}")
        print("------------------------------------------------------")


    # Preload all voxels into cache using fork-share method.
    def _preload_all_voxels(self):
        import glob
        voxel_folder0 = os.path.join(self.voxel_root, "0")  # Currently only using aug=0
        files = glob.glob(os.path.join(voxel_folder0, "*.csv"))
        for fp in tqdm(files, desc="Preloading voxels/0"):
            h = int(os.path.basename(fp).split(".")[0])
            key = (0, h)
            if key in self._cache_world:
                continue
            arr = np.loadtxt(fp, delimiter=",", dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, 3)
            self._cache_world[key] = arr
        
    def  _sample_origin_jitter_world(self) -> np.ndarray:
        # if(not self.is_train) or (self.origin_jitter_ratio <= 0):
        if self.origin_jitter_ratio <= 0:               # For test, also add jitter.
            return np.zeros((3,), dtype=np.float32)
        if self.rng.random() > self.origin_jitter_prob:
            return np.zeros((3,), dtype=np.float32)

        s = float(self.indexer.s)
        a = self.origin_jitter_ratio * s
        jitter = self.rng.uniform(-a, a, size=3).astype(np.float32)     # Add a jitter
        return jitter
    
        

    # ---------------- Length: Use valid_indices ----------------
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    # Load points into cache first. (Maybe not useful now.)     12.04
    def _load_voxel_points_cached(self, aug_idx:int, h:int):
        return self.__load_voxel_points(aug_idx, h)

    # Load normal points into cache first.
    def _load_voxel_normal_points_cached(self, aug_idx: int, h: int, anchor_ijk: tuple[int,int,int]) -> np.ndarray:
        key = (int(aug_idx), int(h))
        if key in self._cache_norm:
            return self._cache_norm[key]

        # 1. Get world coordinates first
        pts_world = self.__load_voxel_points(aug_idx, h)   # Has _cache_world logic inside, can be shared
        # 2. Perform normalization
        pts_norm = self._get_normal_points(pts_world, anchor_ijk)
        # 3. Cache it
        self._cache_norm[key] = pts_norm
        return pts_norm
    
    
    # ---------------- New: Build valid indices per augmentation, expand and save sample info ----------------
    def _build_valid_indices_for_aug(self, aug_idx: int, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Pre-scan a df (corresponding to a specific augmentation aug_idx),
        if the center voxel has points, add the sample to the global list.
        """
        total = len(df)
        num_empty_center = 0
        num_valid = 0

        for idx in tqdm(range(total), desc=f"Pre-scanning aug={aug_idx}"):
            r = df.iloc[idx]
            anchor = (int(r['anchor_i']), int(r['anchor_j']), int(r['anchor_k']))

            # Get 27 neighbor hashes, and take the center voxel hash
            hash_27 = self.indexer.neighbor_hashes(anchor)
            h_center = int(hash_27[self.center_id])

            pts_center = self.__load_voxel_points(aug_idx, h_center)  # np.ndarray [N,3]
            if pts_center.shape[0] > 0:
                # This is a valid sample -> expand and record
                self.aug_idx_list.append(aug_idx)
                self.query_list.append([float(r['x']), float(r['y']), float(r['z'])])
                self.p2v_list.append([float(r['p2v_x']), float(r['p2v_y']), float(r['p2v_z'])])
                self.anchor_list.append([int(anchor[0]), int(anchor[1]), int(anchor[2])])
                num_valid += 1
            else:
                num_empty_center += 1

        return num_valid, num_empty_center
    
    
    # ---------------- Read voxel_points: Add aug_idx dimension ----------------
    def __load_voxel_points(self, aug_idx: int, hash: int) -> np.ndarray:
        key = (int(aug_idx), int(hash))         # Deal with same hash different aug
        if key in self._cache_world:
            return self._cache_world[key]

        voxel_folder = os.path.join(self.voxel_root, str(aug_idx))
        voxel_filename = os.path.join(voxel_folder, f"{int(hash)}.csv")
        if not os.path.isfile(voxel_filename):
            # File doesn't exist, indicating the neighbor voxel is empty
            arr = np.zeros((0, 3), dtype=np.float32)
            self._cache_world[key] = arr
            return arr

        # Use csv reader (slow)
        # df = pd.read_csv(voxel_filename, header=None)
        # voxel_points = df.to_numpy(dtype=np.float32)
        
        # Use pd read
        voxel_points = np.loadtxt(voxel_filename, delimiter=",", dtype=np.float32)
        if voxel_points.ndim == 1:
            voxel_points = voxel_points.reshape(1,3)
        self._cache_world[key] = voxel_points
        return voxel_points
    
    def _sample_or_pad(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = pts.shape[0]
        K = self.n_sample
        
        out = np.zeros((K,3), dtype=np.float32)
        mask = np.zeros((K,), dtype=bool)
        
        if N <= 0:
            return out, mask

        n_vis = 0               # How many points are "visible" for the network
        if not self.is_train:   # Testing, no reset point to zeros
            n_vis = min(N, K)
        else:
            if N < K:           # Not enough voxel points, there must be padding 0 later
                n_vis = N
            else:
                padding_ratio = 0.2
                if np.random.rand() < padding_ratio:
                    assert K >= 1
                    n_vis = np.random.randint(K//2, K)
                else:               # No padding, using full K points
                    n_vis = K 

        if n_vis == N:          # Speed up. Avoid random sampling the whole case
            pts_vis = pts
        else:    
            # Get n_vis points from real points, set others to padding
            idx = np.random.choice(N, n_vis, replace=False)
            pts_vis = pts[idx]

        out[:n_vis] = pts_vis.astype(np.float32, copy=False)
        mask[:n_vis] = True
        
        return out, mask
    

    # Coordinate normalization or back to world
    def _get_anchor_27(self, anchor: tuple[int,int,int]):       # Get neighbor anchors. Anchor is always a tuple.
        i0,j0,k0 = anchor
        arr = np.array([[i0+dx, j0+dy, k0+dj] for (dx,dy,dj) in self.indexer.OFFSET27], dtype=np.int64)
        return arr
    
    # ---------- Coordinate transformation: world <-> normalized local ----------
    # (p-center)/s, range (-0.5, 0.5). 
    def _get_normal_points(self, pts_in_world, anchor: tuple[int,int,int], center_jitter_world: Optional[np.ndarray]=None):
        
        """
        World coordinates -> normalized voxel coordinates ([-0.5,0.5])
        pts_in_world: np.ndarray or torch.Tensor, shape [..., 3]
        anchor: (i,j,k)
        Returns array/tensor of same type/device as input
        """
        center_np = self.indexer.voxel_center(anchor).astype(np.float32)
        if center_jitter_world is not None:
            center_np += center_jitter_world.astype(np.float32)
            
        s = float(self.indexer.s)
        # torch branch
        if isinstance(pts_in_world, torch.Tensor):
            pts = pts_in_world
            center = torch.as_tensor(center_np, dtype=pts.dtype, device=pts.device)
            return (pts - center) / s
        # numpy branch
        pts = np.asarray(pts_in_world, dtype=np.float32)
        center = center_np.astype(np.float32)
        pts_normal = (pts - center) / s
        return pts_normal
    
    # Convert normalized points back to world coordinates
    def _get_world_points(self, pts_norm, anchor: tuple[int,int,int], skip_padding : bool = False):          
        """
        Normalized voxel coordinates ([-0.5,0.5]) -> world coordinates
        pts_norm: np.ndarray or torch.Tensor, shape [..., 3]
        anchor: (i,j,k)
        Returns array/tensor of same type/device as input
        """
        assert np.all((-0.5 <= pts_norm) & (pts_norm <= 0.5)), "pts_norm are not normalized."
        center_np = self.indexer.voxel_center(anchor) # np.ndarray(float64) shape [3]
        s = float(self.indexer.s)
        
        if skip_padding:
            # Create mask to skip points where pts_norm is (0, 0, 0)
            non_zero_pts_mask = np.all(pts_norm != 0, axis=-1)

            # torch branch
            if isinstance(pts_norm, torch.Tensor):
                pts = pts_norm
                center = torch.as_tensor(center_np, dtype=pts.dtype, device=pts.device)
                
                # Skip (0,0,0) points
                pts_world = pts * s + center
                pts_world[~non_zero_pts_mask] = torch.tensor(float('nan'), device=pts.device)  # Or any value to indicate it's skipped
                return pts_world

            # numpy branch
            pts = np.asarray(pts_norm, dtype=np.float32)
            center = center_np.astype(np.float32)

            # Skip (0,0,0) points
            pts_world = pts * s + center
            pts_world[~non_zero_pts_mask] = np.nan  # Or any value to indicate it's skipped
            return pts_world
        
        else:       # Do not skip padding points
            # torch branch
            if isinstance(pts_norm, torch.Tensor):
                pts = pts_norm
                center = torch.as_tensor(center_np, dtype=pts.dtype, device=pts.device)
                return pts * s + center
            # numpy branch
            pts = np.asarray(pts_norm, dtype=np.float32)
            center = center_np.astype(np.float32)
            pts_world = pts * s + center
            return pts_world



    
    def __getitem__(self, idx:int):
        
        # t0 = time.time()
        
        real_idx = self.valid_indices[idx]              # Only use valid index
        aug_idx = int(self.aug_idx_arr[real_idx])       # Which aug-idx for this data
        
        ## Use pre-converted 2025.12.03
        query  = self.query_arr[real_idx]
        p2v    = self.p2v_arr[real_idx]
        p2v    = p2v/self.voxel_size
        anchor = tuple(self.anchor_arr[real_idx])
        center_jitter = self._sample_origin_jitter_world()
        
        
        anchor_27 = self._get_anchor_27(anchor)
        hash_27 = self.indexer.neighbor_hashes(anchor)
        
        # Read 27 voxel points and reshape to [27,K,3]
        points_27 = np.zeros((27, self.n_sample, 3), dtype=np.float32)
        points_mask_27 = np.zeros((27, self.n_sample), dtype=bool)
        exist_mask_27 = np.zeros((27,), dtype=bool)         # Whether the voxel has any points. False: no points in neighbor voxel
        
        for i, h in enumerate(hash_27):
            neighbor_anchor = tuple(anchor_27[i])
            pts_world = self._load_voxel_points_cached(aug_idx, int(h))
            pts = self._get_normal_points(pts_world, neighbor_anchor, center_jitter)
            pts_sample, mask_sample = self._sample_or_pad(pts)
            
            points_27[i] = pts_sample
            points_mask_27[i] = mask_sample
            exist_mask_27[i] = (pts.shape[0] > 0)


        query = torch.from_numpy(self._get_normal_points(query, anchor, center_jitter))
        points_27 = torch.from_numpy(points_27)
        p2v = torch.from_numpy(p2v)
        exist_mask_27 = torch.from_numpy(exist_mask_27)
        points_mask_27 = torch.from_numpy(points_mask_27)
        
        if self.is_train and self.enable_local_rotation:
            points_27, query, p2v = self.apply_local_rotation(points_27, query, p2v, points_mask_27, enable=True)

        sample = {
            'query': query,  # Normalized query, with jitter
            'p2v': p2v,  # [3]
            'points_27': points_27,  # [27,K,3]
            'deltas_27': self.deltas_27,    # [27, 3]: [[-1,-1,-1], [-1,-1,0], ... [1,1,1]]
            'exist_mask_27': exist_mask_27,  # [27]
            'points_mask_27': points_mask_27,  # [27,K]
        }

        
        # Other data for debug
        if self.is_debug:
            arr = np.array(hash_27, dtype=np.uint64)
            low = (arr & 0xFFFFFFFF).astype(np.int64)
            high = ((arr >> 32) & 0xFFFFFFFF).astype(np.int64)
            sample['hash_low'] = low
            sample['hash_high'] = high
            sample['aug_idx'] = aug_idx
            sample['query_world'] = torch.from_numpy(query)  # [3]

            voxel_points = points_27[self.center_id].copy()
            neighbor_voxel_points = np.delete(points_27, self.center_id, axis=0)
            sample['query_world'] = torch.from_numpy(query)  # [3]
            sample['voxel_points'] = torch.from_numpy(voxel_points)  # [K,3]
            sample['neighbor_voxel_points'] = torch.from_numpy(neighbor_voxel_points)  # [26,K,3]
            # Split hash into low/high
            arr = np.array(hash_27, dtype=np.uint64)
            low = (arr & 0xFFFFFFFF).astype(np.int64)
            high = ((arr >> 32) & 0xFFFFFFFF).astype(np.int64)
            sample['hash_low'] = torch.from_numpy(low)
            sample['hash_high'] = torch.from_numpy(high)
            # Not used for training
            sample['delta_ids'] = torch.from_numpy(self.delta_ids) 
            sample['anchor'] = torch.from_numpy(np.array(anchor, dtype=np.int64))
            sample['anchor_27'] = torch.from_numpy(np.array(anchor_27, dtype=np.int64))
            sample['aug_idx'] = torch.tensor(aug_idx, dtype=torch.int64)

        return sample


    def apply_local_rotation(self, points_27, query, p2v, points_mask_27, enable=True):
        """
        Apply SO(3) rotation augmentation to local geometry (without changing voxel topology)
        """
        if not enable:
            return points_27, query, p2v

        device = points_27.device
        dtype  = points_27.dtype

        R = my_utils.random_local_rotation(device, dtype)  # [3,3]

        # points_27: [27, K, 3]
        # Only rotate valid points, padding remains 0
        points_27_rot = points_27 @ R.T
        points_27_rot = points_27_rot * points_mask_27.unsqueeze(-1)
        # query: [3]
        query_rot = query @ R.T
        # p2v: [3]
        p2v_rot = p2v @ R.T

        return points_27_rot, query_rot, p2v_rot
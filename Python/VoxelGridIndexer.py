
import numpy as np
from typing import Iterable, List, Tuple, Dict
import warnings

VOXEL_SIZE = 0.3

class VoxelGridIndexer:
    OFFSET27: List[Tuple[int,int,int]]=[
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    OFFSET2ID: Dict[Tuple[int,int,int], int] = {o: i for i,o in enumerate(OFFSET27)}
    ID2OFFSET: Dict[int, Tuple[int,int,int]] = {i: o for i,o in enumerate(OFFSET27)}
    
    
    def __init__(self, voxel_size: float=VOXEL_SIZE, origin: Iterable[float]=(.0,.0,.0), seed: int = 0x9e3779b97f4a7c15):
        self.s = float(voxel_size)
        self.o = np.asarray(origin, dtype=np.float64)
        self.seed = np.uint64(seed)
        pass
    
    def voxel_index(self, p_world:np.ndarray) -> Tuple[int, int, int]:
        ijk = np.floor((np.asarray(p_world, dtype=np.float64) - self.o) / self.s).astype(np.int64)
        return (int(ijk[0]), int(ijk[1]), int(ijk[2]))
    
    def voxel_center(self, idx: Tuple[int,int,int]) -> np.ndarray:
        i,j,k = idx
        center = self.o + self.s*(np.array([i,j,k]) + 0.5)
        return center
    

    # Hash
    @staticmethod
    # def _zigzag(n:np.int64) -> np.uint64:
        # return np.uint64(np.uint64(n) << np.uint64(1)) ^ np.uint64(np.uint64(n) >> np.uint64(63))
    def _zigzag(n):
        return (n << 1) ^ (n >> 63)
    
    @staticmethod
    def _mix64(x):
        # ignore warnings. This action will "RuntimeWarning: overflow encountered in scalar multiply", but is correct.
        with warnings.catch_warnings():             
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
            x = (x * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
            x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
            x = (x * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
            x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
            return x
        
    def voxel_hash(self, idx:Tuple[int,int,int]) -> int:
        i, j, k = idx
        key = (np.uint64(self._zigzag(np.int64(i))) ^
            (np.uint64(self._zigzag(np.int64(j))) << np.uint64(21)) ^
            (np.uint64(self._zigzag(np.int64(k))) << np.uint64(42)) ^
            np.uint64(self.seed))
        return int(self._mix64(np.uint64(key)))
    

    # Neighbor 
    @staticmethod
    def offset_to_id(dx:int, dy:int, dz:int) -> int:
        return VoxelGridIndexer.OFFSET2ID[(dx,dy,dz)]
    @staticmethod
    def id_to_offset(id:int) -> Tuple[int, int, int]:
        return VoxelGridIndexer.ID2OFFSET[int(id)]
    
    # return full 27 neighbor
    def neighbor_indices(self, anchor: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
        i0, j0, k0 = anchor
        rng = range(-1, 2)
        indices = [(i0+dx, j0+dy, k0+dz) for dx in rng for dy in rng for dz in rng]
        return indices
    def neighbor_hashes(self, anchor: Tuple[int,int,int]) -> list[Tuple[int,int,int]]:
        return [self.voxel_hash(ijk) for ijk in self.neighbor_indices(anchor)]
    
    # return center + 26 neighbor
    def neighbor_indices_sep(self, anchor: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
        i0, j0, k0 = anchor
        center = (i0,j0,k0)
        neigh = []
        rng = range(-1, 2)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    if dx==0 and dy==0 and dz==0:
                        continue
                    neigh.append((i0+dx, j0+dy, k0+dz))
        return center, neigh

    def neighbor_hashes_sep(self, anchor: Tuple[int,int,int]) -> list[Tuple[int,int,int]]:
        center_idx, neigh_idx = self.neighbor_indices_sep(anchor)
        center_hash = self.voxel_hash(center_idx)
        neigh_hash = [self.voxel_hash(ijk) for ijk in neigh_idx]
        return center_hash, neigh_hash

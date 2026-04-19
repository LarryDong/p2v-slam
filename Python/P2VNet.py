import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Iterable, List, Tuple, Dict
import numpy as np


# ---------- Fourier positional encoding (allow out-of-bounds, no clamping) ----------
def fourier_pe_u(u_local: torch.Tensor) -> torch.Tensor:
    """
    u_local: [B,3] allows exceeding [-0.5,0.5] (e.g., after jitter to [-0.75,0.75])
    Returns: [B,21] = [u, sin/cos(1,2,4)*2pi]
    """
    B = u_local.shape[0]
    freqs = torch.tensor([1.0, 2.0, 4.0], device=u_local.device, dtype=u_local.dtype) * (2 * math.pi)
    ang = u_local.unsqueeze(-1) * freqs
    sin = torch.sin(ang)
    cos = torch.cos(ang)
    pe = torch.cat([sin, cos], dim=-1).reshape(B, -1)
    return torch.cat([u_local, pe], dim=-1)


########################################################################

class MLP_feat(nn.Module):          # 3 -> (128, 64) -> 64
    def __init__(self, in_channels=3, output_dim=64, dropout=0.01):
        
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self._init_weights()        # Initialize weights
        
    def _init_weights(self):
        """Initialize weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)


    @staticmethod
    def _masked_max(x, mask):       # max-pool
        """Masked max pooling
        Args:
            x: [B, K, C] input features
            mask: [B, K] mask, True indicates valid points
        Returns:
            [B, C] pooled features
        """
        if mask is None:
            return x.max(dim=1).values

        # Fill invalid points with negative infinity
        # mask.unsqueeze(-1) expands to [B, K, 1] to match x dimensions
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))    
        
        # Take max along K dimension
        m = x_masked.max(dim=1).values
        
        # Replace negative infinity with 0
        feat = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        return feat
    
    def forward(self, pts:torch.Tensor, mask=None):
        """Forward pass
        Args:
            points: [B, K, C] input point cloud, K=24, C=3
            mask: [B, K] mask, True indicates valid points
        Returns:
            [B, feat_dim] output features, feat_dim=64
        """
        
        B,K,C = pts.shape
        points_flat = pts.view(-1,C)
        tmp = pts.reshape(B*K, -1)
        
        # layer 1: 3->128
        x = F.relu(self.fc1(points_flat))
        x = F.dropout(x, self.dropout)        # if need dropout
        
        # layer 2: 128->64
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout)
        
        # output layer: 64->64
        x = self.fc3(x)
    
        # Reshape back to [B, K, 64] for pooling
        feat = x.view(B, K, -1)  # [B, K, 64]
        
        # Apply masked max pooling to aggregate features from K points
        # -> First extract features, then set masked point features to -inf, then pooling ignores those inputs
        feat = self._masked_max(feat, mask)  # 

        return feat



class MLP_score(nn.Module):
    def  __init__(self, in_dim=94, dropout=0.01):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)        # Normalize 3 input features
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = dropout
        
    def forward(self, z, exist_mask):
        """Forward pass
        Args:
            z: [B, 27, 94] concatenated features (feat+u_pe+embedding)
            exist_mask: [B, 27] existence mask, True indicates valid
        Returns:
            alpha: [B, 27] normalized weights (softmax)
            logits: [B, 27] unnormalized scores
        """

        x = self.norm(z)
        
        # layer 1: 94->ReLU->128
        x = F.relu(self.fc1(x), inplace=False)  # [B, 27, 128]
        x = F.dropout(x, p=self.dropout)
        
        # layer 2: 128->ReLU->64
        x = F.relu(self.fc2(x), inplace=False)  # [B, 27, 64]
        x = F.dropout(x, p=self.dropout)
        
        # Output layer: 64 -> 1, no activation function
        logits = self.out(x).squeeze(-1)  # [B, 27]
        
        # Apply mask: set invalid positions to negative infinity
        logits = logits.masked_fill(~exist_mask.bool(), -1e4)
        
        # Softmax normalization to get weights
        alpha = F.softmax(logits, dim=1)  # [B, 27]
        
        if torch.isnan(alpha).any():
            print("GeoAware nan!")
            nan_rows = torch.isnan(alpha).any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            print("nan_rows:", nan_rows)
            print("exist_mask[nan_rows]:", exist_mask[nan_rows])
            print("logits[nan_rows]:", logits[nan_rows])

        return alpha, logits


# decoder for p2v vector
class MLP_delta(nn.Module):
    def __init__(self, in_dim=94, dropout=0.01):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)   
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 3)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=False)
        x = F.dropout(x,p=self.dropout)
        
        x = F.relu(self.fc2(x), inplace=False)
        x = F.dropout(x,p=self.dropout)

        vec = self.output(x)
        return  vec


# decoder for p2v vector
class MLP_sigma(nn.Module):
    def __init__(self, in_dim=94, dropout=0.01):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)   
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=False)
        x = F.dropout(x,p=self.dropout)
        
        x = F.relu(self.fc2(x), inplace=False)
        x = F.dropout(x,p=self.dropout)
        
        sigma = self.output(x)
        sigma = F.softplus(sigma)       # To avoid negative
        return  sigma




###########################################################################
##########   P2V Model
###########################################################################

class P2VModel(nn.Module):
    def __init__(self, feat_dim=64, u_dim=21, dropout=0.01):
        super().__init__()
        self.feat_dim = feat_dim
        self.upe_dim = u_dim
        self.point_channels = 3              # [x,y,z] for each point
        self.embedding_dim = 9          # embedding: [delta, |delta|, delta^2]
        self.phi_dim = self.feat_dim + self.embedding_dim + self.upe_dim
        
        self.mlp_feat = MLP_feat(in_channels=self.point_channels, output_dim=self.feat_dim, dropout=dropout)           # MLP_feat: 3->64
        self.mlp_score = MLP_score(in_dim=self.phi_dim, dropout=dropout)               # MLP_score: Phi -> Score
        self.mlp_delta = MLP_delta(in_dim=self.phi_dim, dropout=dropout)                  # MLP_delta: Phi -> delta(3)
        self.mlp_sigma = MLP_sigma(in_dim=self.phi_dim, dropout=dropout)                  # MLP_sigma: Phi -> sigma(1)

        self.deltas_27 = torch.tensor([
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ], dtype=torch.float32)  # Directly use float32 to avoid later conversion
        print('------------ P2V Model Inited ------------')
        print(f'feat dim: {self.feat_dim}')
        print(f'u_pe dim: {self.upe_dim}')
        print(f'phi  dim: {self.phi_dim}')
        print(f'dropout : {dropout}')
        print('------------------------------------------')


    def forward(self, points_27, u, exist_mask_27, points_mask_27, return_center_feat=False):
        B, Nn, K, C = points_27.shape
        assert Nn == 27, "Must be 27 voxels! Wrong dim."

        pts_flat = points_27.view(B*Nn, K, C)
        mask_flat = points_mask_27.view(B*Nn, K)

        # MLP_feature
        feat_flat = self.mlp_feat(pts_flat, mask_flat)                 # [B*K, feat_dim]
        feat = feat_flat.view(B, Nn, self.feat_dim)                    # Reshape back. [B, Nn, feat_dim]
        
        # P2V decoder parts.
        # Concatenate all implicit features.
        delta = u.unsqueeze(1) - self.deltas_27.to(u.device)          # Optimize: use fixed value, and the same device
        voxel_embedding = torch.cat([delta, delta.abs(), delta**2], dim=-1)
        u_pe = fourier_pe_u(u)
        u_rep = u_pe.unsqueeze(1).expand(B, Nn, -1)
        
        z = torch.cat([feat, u_rep, voxel_embedding], dim=-1)          # [B, Nn, phi_dim]
        
        score, _ = self.mlp_score(z, exist_mask_27)
        phi = (score.unsqueeze(-1) * z).sum(dim=1)   # [B, D]
        
        # Decode for p2v vector and uncertainty
        p2v_vec = self.mlp_delta(phi)
        p2v_sigma = self.mlp_sigma(phi)
        
        if torch.isnan(p2v_vec).sum().item() > 0:
            print("Nan detected!")
        
        if not return_center_feat:
            return p2v_vec, p2v_sigma, score, phi
        else:
            center_id = 13
            return feat[:, center_id, :]


###############################################
#   Joint training of VE_Net and P2V_Net
###############################################
class JointTrainigModel(nn.Module):
    def __init__(self, feat_dim=64, u_dim=21, dropout=0.01):
        super().__init__()
        self.feat_dim = feat_dim
        self.upe_dim = u_dim
        self.embedding_dim = 9          # embedding: [delta, |delta|, delta^2]
        self.phi_dim = self.feat_dim + self.embedding_dim + self.upe_dim
        
        self.ve_net = VE_Net(feat_dim=feat_dim, u_dim=u_dim, dropout=dropout)
        self.p2v_net = P2V_Net(feat_dim=feat_dim, u_dim=u_dim, dropout=dropout)
        print('-> JointTrainingModel inited.')
        
    def forward(self, points_27, u, exist_mask_27, points_mask_27, return_center_feat=False):
        B, Nn, K, C = points_27.shape
        
        pts_flat = points_27.view(B*Nn, K, C)
        mask_flat = points_mask_27.view(B*Nn, K)

        # VE_Net, extract feature 
        feat_flat = self.ve_net(pts_flat, mask_flat)                 # [B*K, feat_dim]
        feat27 = feat_flat.view(B, Nn, self.feat_dim)                    # Reshape back. [B, Nn, feat_dim]
        
        # P2V_Net, predict feature
        p2v_vec, p2v_sigma, score, phi = self.p2v_net(feat27, u, exist_mask_27)

        if not return_center_feat:
            return p2v_vec, p2v_sigma, score, phi
        else:
            center_id = 13
            return feat27[:, center_id, :]



# Split Model

class VE_Net(nn.Module):
    def __init__(self, feat_dim=64, u_dim=21, dropout=0.01):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp_feat = MLP_feat(in_channels=3, output_dim=self.feat_dim, dropout=dropout)
        print(f"-> VE_Net inited.")
        
    def forward(self, pts, masks):
        return self.mlp_feat(pts, masks)

class P2V_Net(nn.Module):
    def __init__(self, feat_dim=64, u_dim=21, dropout=0.01):
        super().__init__()
        self.feat_dim = feat_dim
        self.upe_dim = u_dim
        self.embedding_dim = 9
        self.phi_dim = self.feat_dim + self.embedding_dim + self.upe_dim
        
        self.mlp_score = MLP_score(in_dim=self.phi_dim, dropout=dropout)
        self.mlp_delta = MLP_delta(in_dim=self.phi_dim, dropout=dropout)
        self.mlp_sigma = MLP_sigma(in_dim=self.phi_dim, dropout=dropout)
        
        self.deltas_27 = torch.tensor([
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ], dtype=torch.float32)  # Directly use float32 to avoid later conversion
        
        print(f"-> P2V_Net inited.")


    def forward(self, feat27, u, exist_mask_27):
        B, Nn, _ = feat27.shape
        u_pe = fourier_pe_u(u)
        u_rep = u_pe.unsqueeze(1).expand(B, Nn, -1)
        delta = u.unsqueeze(1) - self.deltas_27.to(u.device)          # Optimize: use fixed value, and the same device
        voxel_embedding = torch.cat([delta, delta.abs(), delta**2], dim=-1)
        
        z = torch.cat([feat27, u_rep, voxel_embedding], dim=-1)          # [B, Nn, phi_dim]
        
        score, _ = self.mlp_score(z, exist_mask_27)
        phi = (score.unsqueeze(-1) * z).sum(dim=1)   # [B, D]
        
        # Decode for p2v vector and uncertainty
        p2v_vec = self.mlp_delta(phi)
        p2v_sigma = self.mlp_sigma(phi)
        
        return p2v_vec, p2v_sigma, score, phi


def loss_p2v(pred_p2v, gt_p2v, sigma_d, 
             mode='vec',            # vec, or dir_dist
             # ===== Common hyperparameters =====
             eps=1e-9,
             # ===== vec mode hyperparameters =====
             beta_vec=0.01,
             # ===== dir_dist mode hyperparameters =====
             lambda_dir=0.1,
             beta_dist=0.01,
             tau_dir=0.0,
             use_nll=False,
             weight_z=False):           # If p2v_gt length < tau_dir, ignore direction.

    # --------- Mode 1: vec mode (baseline) ---------
    if mode == 'vec':
        # Vector channel-wise Smooth L1
        per = F.smooth_l1_loss(pred_p2v, gt_p2v, beta=beta_vec, reduction='none')  # [B,3]

        if weight_z:  # weight_z: bool
            wz = 1.5
            w = pred_p2v.new_tensor([1.0, 1.0, wz]).view(1, 3)  # wz: float, e.g. 1.2
            per = per * w

        loss_vec = per.sum(dim=-1, keepdim=True).mean()  # scalar
        
        if use_nll:
            lambda_d = 0.01          # TODO: Using different nll weight.
            d_gt = torch.linalg.norm(gt_p2v, dim=-1, keepdim=True).clamp_min(eps)
            d_pred = torch.linalg.norm(pred_p2v, dim=-1, keepdim=True).clamp_min(eps)
            sigma2 = (sigma_d**2).clamp_min(eps)
            diff2 = (d_pred - d_gt) ** 2
            loss_d = 0.5 * (diff2/sigma2 + torch.log(sigma2))
            loss_per = loss_vec + lambda_d * loss_d
            loss_tot = loss_per.mean()
            loss_d_mean = loss_d.mean()
            return loss_tot, loss_vec, loss_d_mean
        
        else:
            loss_tot = loss_vec
            return loss_tot, loss_vec, loss_vec

    # --------- Mode 2: direction + distance mode ---------
    elif mode == 'dir_dist':
        # --------- Common: magnitude calculation ---------
        d_gt   = torch.linalg.norm(gt_p2v,   dim=-1, keepdim=True).clamp_min(eps)  # [B,1]
        d_pred = torch.linalg.norm(pred_p2v, dim=-1, keepdim=True).clamp_min(eps)  # [B,1]

        # Unit direction
        dir_gt   = gt_p2v   / d_gt                                     # [B,3]
        dir_pred = pred_p2v / d_pred                                   # [B,3]

        # Direction cosine loss: 1 - cos(theta)
        cos_sim = (dir_pred * dir_gt).sum(dim=-1, keepdim=True)        # [B,1]
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        loss_dir = 1.0 - cos_sim                                       # [B,1]

        # Optional: For very small magnitude gt, don't train direction
        if tau_dir is not None and tau_dir > 0.0:
            mask = (d_gt > tau_dir).float()
            loss_dir = loss_dir * mask

        # Distance SmoothL1 (only magnitude)
        loss_dist = F.smooth_l1_loss(d_pred, d_gt, beta=beta_dist, reduction='none') # [B,1]

        # Main loss = direction + distance
        loss_tot = loss_dist + lambda_dir * loss_dir    # [B,1]
        
        loss_tot = loss_tot.mean()
        loss_dist = loss_dist.mean()
        
        # Angle error. Only for visualization
        angle_err_deg = torch.sqrt(torch.clamp(2.0 * loss_dir, min=0.0) + eps) * 180/math.pi
        angle_err_deg = angle_err_deg.mean()
        loss_dir = loss_dir.mean()
        
        return loss_tot, loss_dist, angle_err_deg
    
    else:
        print(f"Invalid mode: {mode}")
        raise ValueError('Invalid mode: {}'.format(mode))



def test_specific_vectors():
    """
    Test specific vectors (0,0,1) and (1,0,0)
    """
    print("=" * 60)
    print("Testing dir_dist loss function - Specific vector examples")
    print("=" * 60)
    
    # Create specific tensors
    # Case 1: pred=(0,0,1), gt=(1,0,0) - orthogonal vectors
    print("\nCase 1: Orthogonal vectors")
    print("Pred: (0, 0, 1)")
    print("GT: (1, 0, 0)")
    
    pred_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    gt_p2v = torch.tensor([[1.0, 0.0, 0.0]])
    sigma_d = 1.0
    
    loss_tot, loss_dist, angle_err_deg = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0
    )
    
    print(f"Total loss: {loss_tot.item():.6f}")
    print(f"Distance loss: {loss_dist.item():.6f}")
    print(f"Angle error: {angle_err_deg.item():.6f}°")
    print(f"Theoretical angle: 90° (dot product is 0)")
    print(f"Expected: Angle error should be close to 90 degrees")
    
    # Case 2: pred=(0,0,1), gt=(0,0,2) - same direction, different magnitude
    print("\n" + "-" * 40)
    print("Case 2: Same direction, different magnitude")
    print("Pred: (0, 0, 1)")
    print("GT: (0, 0, 2)")
    
    pred_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    gt_p2v = torch.tensor([[0.0, 0.0, 2.0]])
    
    loss_tot, loss_dist, angle_err_deg = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0
    )
    
    print(f"Total loss: {loss_tot.item():.6f}")
    print(f"Distance loss: {loss_dist.item():.6f}")
    print(f"Angle error: {angle_err_deg.item():.6f}°")
    print(f"Expected: Angle error should be close to 0 degrees, distance loss should be large")
    
    # Case 3: pred=(0,0,1), gt=(0,0,1) - exactly the same
    print("\n" + "-" * 40)
    print("Case 3: Exactly the same")
    print("Pred: (0, 0, 1)")
    print("GT: (0, 0, 1)")
    
    pred_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    gt_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    
    loss_tot, loss_dist, angle_err_deg = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0
    )
    
    print(f"Total loss: {loss_tot.item():.6f}")
    print(f"Distance loss: {loss_dist.item():.6f}")
    print(f"Angle error: {angle_err_deg.item():.6f}°")
    print(f"Expected: All losses should be close to 0")
    
    # Case 4: pred=(0,0,1), gt=(0,0,-1) - opposite direction
    print("\n" + "-" * 40)
    print("Case 4: Opposite direction")
    print("Pred: (0, 0, 1)")
    print("GT: (0, 0, -1)")
    
    pred_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    gt_p2v = torch.tensor([[0.0, 0.0, -1.0]])
    
    loss_tot, loss_dist, angle_err_deg = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0
    )
    
    print(f"Total loss: {loss_tot.item():.6f}")
    print(f"Distance loss: {loss_dist.item():.6f}")
    print(f"Angle error: {angle_err_deg.item():.6f}°")
    print(f"Theoretical angle: 180° (dot product is -1)")
    
    # Case 5: Batch test
    print("\n" + "=" * 40)
    print("Case 5: Batch test")
    
    batch_pred = torch.tensor([
        [0.0, 0.0, 1.0],    # Orthogonal to first gt
        [0.0, 0.0, 1.0],    # Same as second gt
        [0.0, 0.0, 1.0],    # Opposite to third gt
    ])
    
    batch_gt = torch.tensor([
        [1.0, 0.0, 0.0],    # Orthogonal
        [0.0, 0.0, 1.0],    # Same
        [0.0, 0.0, -1.0],   # Opposite
    ])
    
    loss_tot, loss_dist, angle_err_deg = loss_p2v(
        batch_pred, batch_gt, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0
    )
    
    print(f"Batch total loss: {loss_tot.item():.6f}")
    print(f"Batch distance loss: {loss_dist.item():.6f}")
    print(f"Batch angle error: {angle_err_deg.item():.6f}°")
    print("Note: Angle error is batch average")
    
    # Case 6: Test tau_dir parameter
    print("\n" + "=" * 40)
    print("Case 6: Testing tau_dir parameter (tau_dir=1.5)")
    
    pred_p2v = torch.tensor([[0.0, 0.0, 1.0]])
    gt_p2v = torch.tensor([[0.0, 1.0, 1.0]])  # Magnitude = 1 < 1.5
    
    # Without using tau_dir
    loss_tot1, loss_dist1, angle_err_deg1 = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=0.0  # No restriction
    )
    
    # Using tau_dir
    loss_tot2, loss_dist2, angle_err_deg2 = loss_p2v(
        pred_p2v, gt_p2v, sigma_d,
        mode='dir_dist',
        lambda_dir=0.1,
        beta_dist=0.01,
        tau_dir=1.5  # Ignore direction loss
    )
    
    print(f"Without tau_dir - Total loss: {loss_tot1.item():.6f}, Angle error: {angle_err_deg1.item():.6f}°")
    print(f"With tau_dir=1.5 - Total loss: {loss_tot2.item():.6f}, Angle error: {angle_err_deg2.item():.6f}°")
    print("Note: When true vector magnitude is less than tau_dir, direction loss should be ignored")

if __name__ == "__main__":
    # Make sure loss_p2v function is defined above
    # You can copy the loss_p2v function provided above here
    test_specific_vectors()
    
    
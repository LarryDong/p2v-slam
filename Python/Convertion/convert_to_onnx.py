
# Convert a joint-pth model into two seperated .onnx model (ir-net and ve-net)

import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import P2VNet

class Config:
    # Model parameters
    feat_dim = 64
    u_dim = 21
    dropout = 0.0
    device = "cpu"
    
    # File paths
    model_base_folder = "../Output/model_output"            # Your output folder after training.
    model_name = "joint-model"                                    # Your model name
    
    @property
    def full_model_pth(self):
        return f"{self.model_base_folder}/{self.model_name}.pth"
    
    @property
    def encoder_pth(self):
        return f"{self.model_base_folder}/ve-net-python.pth"
    
    @property
    def decoder_pth(self):
        return f"{self.model_base_folder}/ir-net-python.pth"
    
    @property
    def encoder_onnx(self):
        return f"{self.model_base_folder}/ve-net.onnx"
    
    @property
    def decoder_onnx(self):
        return f"{self.model_base_folder}/ir-net.onnx"


class P2VNetDecoderWrapper(torch.nn.Module):
    """
    Wrapper for P2V_Net to make ONNX export stable.
    Registers deltas_27 and fourier frequencies as buffers.
    """
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        
        # Register deltas_27 as buffer if it exists
        if hasattr(decoder, 'deltas_27'):
            self.register_buffer('deltas_27', decoder.deltas_27.clone())
        
        # Register fourier frequencies
        freqs = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32) * (2.0 * np.pi)
        self.register_buffer('fourier_freqs', freqs)
    
    def forward(self, feat27, u, exist_mask_27):
        B, Nn, _ = feat27.shape
        
        # Position encoding for u
        ang = u.unsqueeze(-1) * self.fourier_freqs  # [B, 3, 3]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        pe = torch.cat([sin, cos], dim=-1).reshape(B, -1)  # [B, 18]
        u_pe = torch.cat([u, pe], dim=-1)  # [B, 21]
        
        # Repeat for all neighbor voxels
        u_rep = u_pe.unsqueeze(1).expand(B, Nn, -1)  # [B, 27, 21]
        
        # Voxel embedding with deltas
        deltas = self.deltas_27.to(u.device, dtype=u.dtype)
        delta = u.unsqueeze(1) - deltas  # [B, 27, 3]
        voxel_embedding = torch.cat([delta, delta.abs(), delta**2], dim=-1)  # [B, 27, 9]
        
        # Concatenate all features
        z = torch.cat([feat27, u_rep, voxel_embedding], dim=-1)  # [B, 27, 94]
        
        # Forward through MLPs
        score, _ = self.decoder.mlp_score(z, exist_mask_27)
        phi = (score.unsqueeze(-1) * z).sum(dim=1)  # [B, 94]
        
        vec = self.decoder.mlp_delta(phi)  # [B, 3]
        sigma = self.decoder.mlp_sigma(phi)  # [B, 1]
        
        return vec, sigma, score, phi


def split_full_model(config: Config):
    """Split full model into encoder and decoder .pth files"""
    print(f"Splitting full model: {config.full_model_pth}")
    
    # Load full model
    model_full = P2VNet.JointTrainigModel(
        feat_dim=config.feat_dim, 
        u_dim=config.u_dim, 
        dropout=config.dropout
    )
    ckpt = torch.load(config.full_model_pth, map_location=config.device)
    state = ckpt.get("model_state_dict", ckpt)
    model_full.load_state_dict(state, strict=True)
    model_full.eval()
    
    # Extract encoder (VE_Net)
    encoder = P2VNet.VE_Net(
        feat_dim=config.feat_dim, 
        u_dim=config.u_dim, 
        dropout=config.dropout
    )
    encoder.load_state_dict(model_full.ve_net.state_dict(), strict=True)
    
    # Extract decoder (P2V_Net)
    decoder = P2VNet.P2V_Net(
        feat_dim=config.feat_dim, 
        u_dim=config.u_dim, 
        dropout=config.dropout
    )
    decoder.load_state_dict(model_full.p2v_net.state_dict(), strict=True)
    
    # Save split models
    torch.save(encoder.state_dict(), config.encoder_pth)
    torch.save(decoder.state_dict(), config.decoder_pth)
    
    print(f"  ✓ Encoder saved to: {config.encoder_pth}")
    print(f"  ✓ Decoder saved to: {config.decoder_pth}\n")


def export_encoder_to_onnx(config: Config):
    """Export encoder to ONNX"""
    print(f"Exporting encoder to ONNX: {config.encoder_onnx}")
    
    # Load encoder
    encoder = P2VNet.VE_Net(
        feat_dim=config.feat_dim, 
        u_dim=config.u_dim, 
        dropout=config.dropout
    )
    encoder.load_state_dict(torch.load(config.encoder_pth, map_location=config.device))
    encoder.eval().to(config.device)
    
    # Dummy input
    B, K = 1, 40
    dummy_pts = torch.randn(B, K, 3, dtype=torch.float32, device=config.device)
    dummy_mask = torch.ones(B, K, dtype=torch.bool, device=config.device)
    
    # Export
    torch.onnx.export(
        encoder,
        (dummy_pts, dummy_mask),
        config.encoder_onnx,
        input_names=["pts", "mask"],
        output_names=["feat"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "pts": {0: "batch"},
            "mask": {0: "batch"},
            "feat": {0: "batch"},
        }
    )
    
    onnx.checker.check_model(onnx.load(config.encoder_onnx))
    print(f"  ✓ Encoder ONNX saved\n")


def export_decoder_to_onnx(config: Config):
    """Export decoder to ONNX"""
    print(f"Exporting decoder to ONNX: {config.decoder_onnx}")
    
    # Load decoder
    decoder = P2VNet.P2V_Net(
        feat_dim=config.feat_dim, 
        u_dim=config.u_dim, 
        dropout=config.dropout
    )
    decoder.load_state_dict(torch.load(config.decoder_pth, map_location=config.device))
    decoder.eval().to(config.device)
    
    # Wrap for stable export
    decoder_wrapped = P2VNetDecoderWrapper(decoder).to(config.device).eval()
    
    # Dummy inputs
    B, Nn, feat_dim = 1, 27, config.feat_dim
    dummy_feat27 = torch.randn(B, Nn, feat_dim, dtype=torch.float32, device=config.device)
    dummy_u = torch.randn(B, 3, dtype=torch.float32, device=config.device)
    dummy_exist_mask = torch.ones(B, Nn, dtype=torch.bool, device=config.device)
    
    # Export
    torch.onnx.export(
        decoder_wrapped,
        (dummy_feat27, dummy_u, dummy_exist_mask),
        config.decoder_onnx,
        input_names=["feat27", "u", "exist_mask_27"],
        output_names=["vec", "sigma", "score", "phi"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "feat27": {0: "batch"},
            "u": {0: "batch"},
            "exist_mask_27": {0: "batch"},
            "vec": {0: "batch"},
            "sigma": {0: "batch"},
            "score": {0: "batch"},
            "phi": {0: "batch"},
        }
    )
    
    onnx.checker.check_model(onnx.load(config.decoder_onnx))
    print(f"  ✓ Decoder ONNX saved\n")


def verify_onnx(config: Config):
    """Verify exported ONNX models produce same outputs as PyTorch"""
    print("Verifying ONNX exports...")
    
    # Load PyTorch models
    encoder = P2VNet.VE_Net(feat_dim=config.feat_dim, u_dim=config.u_dim, dropout=config.dropout)
    encoder.load_state_dict(torch.load(config.encoder_pth, map_location=config.device))
    encoder.eval()
    
    decoder = P2VNet.P2V_Net(feat_dim=config.feat_dim, u_dim=config.u_dim, dropout=config.dropout)
    decoder.load_state_dict(torch.load(config.decoder_pth, map_location=config.device))
    decoder.eval()
    
    # Load ONNX models
    ort_session_enc = ort.InferenceSession(config.encoder_onnx, providers=["CPUExecutionProvider"])
    ort_session_dec = ort.InferenceSession(config.decoder_onnx, providers=["CPUExecutionProvider"])
    
    # Dummy inputs
    B, K, Nn = 1, 40, 27
    dummy_pts = torch.randn(B, K, 3, dtype=torch.float32)
    dummy_mask = torch.ones(B, K, dtype=torch.bool)
    dummy_feat27 = torch.randn(B, Nn, config.feat_dim, dtype=torch.float32)
    dummy_u = torch.randn(B, 3, dtype=torch.float32)
    dummy_exist = torch.ones(B, Nn, dtype=torch.bool)
    
    # PyTorch inference
    with torch.no_grad():
        torch_feat = encoder(dummy_pts, dummy_mask)
        torch_vec, torch_sigma, torch_score, torch_phi = decoder(dummy_feat27, dummy_u, dummy_exist)
    
    # ONNX inference
    onnx_feat = ort_session_enc.run(["feat"], {"pts": dummy_pts.numpy(), "mask": dummy_mask.numpy()})[0]
    onnx_vec, onnx_sigma, onnx_score, onnx_phi = ort_session_dec.run(
        ["vec", "sigma", "score", "phi"],
        {"feat27": dummy_feat27.numpy(), "u": dummy_u.numpy(), "exist_mask_27": dummy_exist.numpy()}
    )
    
    # Compare
    def compare(name, torch_out, onnx_out, atol=1e-5):
        diff = np.max(np.abs(torch_out.numpy() - onnx_out))
        ok = diff < atol
        print(f"  {name}: {'✓' if ok else '✗'} max_diff={diff:.3e}")
        return ok
    
    print("\n  Encoder output:")
    compare("feat", torch_feat, onnx_feat)
    
    print("  Decoder outputs:")
    compare("vec", torch_vec, onnx_vec)
    compare("sigma", torch_sigma, onnx_sigma)
    compare("score", torch_score, onnx_score)
    compare("phi", torch_phi, onnx_phi)
    print()


def main():
    config = Config()
    
    # Step 1: Split full model into encoder and decoder .pth files
    split_full_model(config)
    
    # Step 2: Convert to ONNX
    export_encoder_to_onnx(config)
    export_decoder_to_onnx(config)
    
    # Step 3: Verify correctness (optional)
    verify_onnx(config)
    
    print("✅ All done! Full model split and converted to ONNX successfully.")
    print(f"   Encoder: {config.encoder_onnx}")
    print(f"   Decoder: {config.decoder_onnx}")


if __name__ == "__main__":
    main()
    



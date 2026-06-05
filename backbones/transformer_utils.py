import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # -> [B, embed_dim, H//P, W//P]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # -> [B, N, C]
        return x, H, W

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.reproj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        # x: [B, N, C] �?[B, C, H, W]
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        x = self.reproj(x)  # 恢复回原图尺�?
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, C] �?where N = H*W, C = channels
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out  # Residual after attention

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual  # Residual after FFN
        return x

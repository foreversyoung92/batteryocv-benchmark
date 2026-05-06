"""
models.py
=========
Discriminative architectures for high-to-low rate profile reconstruction.

Common forward interface:
    forward(x_main, x_aux) -> (y_prof, y_cap, z)
        x_main : [B, W, 128]       W = window_size, profile_len = 128
        x_aux  : [B, W, aux_dim]   aux_dim depends on the variant
        y_prof : [B, 1, 128]
        y_cap  : [B, 2]
        z      : [B, latent_dim]

Auxiliary fusion strategies:
    Conv / MLP / UNet : late fusion  (aux encoded separately, concat to latent)
    LSTM / BiLSTM     : early fusion (aux concatenated at each time step)
    Transformer       : condition token (CLS + aux embedding)
    ConvLSTM hybrid   : Conv profile encoder + LSTM over the window sequence

Architectures:
    "conv"        Conv1d AE
    "mlp"         MLP AE
    "lstm"        LSTM AE
    "bilstm"      Bidirectional LSTM AE
    "transformer" Transformer AE
    "unet"        1D UNet AE
    "conv_lstm"   Conv + LSTM hybrid AE
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =========================================================
# Common utilities
# =========================================================
class SegmentBias(nn.Module):
    """
    Learnable per-segment bias for charge[0:64] and discharge[64:128].
    Broadcast over the last dim (profile_len = 128) regardless of input rank.
    Works for shapes [B, W, L], [B, C, L], [B*W, 1, L], etc.
    """
    def __init__(self, length: int = 128, split: int = 64):
        super().__init__()
        assert split * 2 == length
        self.seg = nn.Parameter(torch.zeros(2, split))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat([self.seg[0], self.seg[1]], dim=0)  # [128]
        view_shape = (1,) * (x.dim() - 1) + (bias.shape[0],)
        return x + bias.view(view_shape)


def _cap_head(latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
        nn.Linear(latent_dim // 2, 2),
    )


def _out_act(name: str) -> nn.Module:
    return {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}.get(name, nn.Identity())


def _aux_late_fusion(aux_dim: int, window_size: int, aux_embed_dim: int, latent_dim: int):
    """Build a small module bundle that handles aux as late fusion."""
    aux_encoder = nn.Sequential(
        nn.Linear(aux_dim, aux_embed_dim), nn.ReLU(),
        nn.Linear(aux_embed_dim, aux_embed_dim), nn.ReLU(),
    )
    aux_to_latent = nn.Sequential(
        nn.Linear(window_size * aux_embed_dim, latent_dim // 2), nn.ReLU(),
    )
    return aux_encoder, aux_to_latent


def _flatten_channel_as_conv(x: torch.Tensor) -> torch.Tensor:
    """[B,W,C,L] -> [B,W*C,L], [B,W,L] unchanged."""
    if x.dim() == 4:
        b, w, c, l = x.shape
        return x.reshape(b, w * c, l)
    return x


def _flatten_channel_as_token(x: torch.Tensor) -> torch.Tensor:
    """[B,W,C,L] -> [B,W,C*L], [B,W,L] unchanged."""
    if x.dim() == 4:
        b, w, c, l = x.shape
        return x.reshape(b, w, c * l)
    return x


def _cat_aux(x: torch.Tensor, x_aux: Optional[torch.Tensor]) -> torch.Tensor:
    if x_aux is None or x_aux.numel() == 0 or x_aux.shape[-1] == 0:
        return x
    return torch.cat([x, x_aux], dim=-1)


def _conv_flat_len(profile_len: int) -> int:
    return max(1, math.ceil(profile_len / 8))


# =========================================================
# 1. Conv1d AE (baseline)
# =========================================================
class ConvAE(nn.Module):
    """
    Conv1d encoder/decoder + late fusion aux
    Encoder: [B, W, 128] → Conv1d → [B, latent_dim]
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        aux_embed_dim: int = 8,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.total_in_ch = window_size * in_channels
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        # Encoder: Conv1d treats W as input channels and slides along profile_len.
        self.encoder = nn.Sequential(
            nn.Conv1d(self.total_in_ch, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 128→64
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),          nn.ReLU(),  # 64→32
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),          nn.ReLU(),  # 32→16
        )
        enc_len = _conv_flat_len(profile_len)
        enc_dim = 16 * enc_len
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(enc_dim, latent_dim), nn.ReLU())

        # aux late fusion
        self.aux_enc, self.aux_to_lat = _aux_late_fusion(aux_dim, window_size, aux_embed_dim, latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim), nn.ReLU(),
        )

        # decoder
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, enc_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64,  1, 3, stride=2, padding=1, output_padding=1),
        )

        self.out_act = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)
        x = self.seg_bias(_flatten_channel_as_conv(x_main))
        z_main = self.to_latent(self.encoder(x))

        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            z_aux = self.aux_to_lat(self.aux_enc(x_aux).reshape(b, -1))
            z = self.fusion(torch.cat([z_main, z_aux], dim=1))
        else:
            z = z_main

        h = self.from_latent(z).view(b, 16, _conv_flat_len(x_main.shape[-1]))
        y_prof = self.out_act(self.decoder(h))
        if y_prof.shape[-1] != x_main.shape[-1]:
            y_prof = F.interpolate(y_prof, size=x_main.shape[-1], mode="linear", align_corners=False)
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 2. MLP AE
# =========================================================
class MLPAE(nn.Module):
    """
    Flatten the window, then MLP encoder/decoder with late-fusion aux.
    Simplest baseline; serves as a lower-bound reference.
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        aux_embed_dim: int = 8,
        hidden_dim: int = 512,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.profile_len = profile_len
        self.in_channels = in_channels
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        input_dim = window_size * in_channels * profile_len

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim), nn.ReLU(),
        )

        # aux late fusion
        self.aux_enc, self.aux_to_lat = _aux_late_fusion(aux_dim, window_size, aux_embed_dim, latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, profile_len),
        )

        self.out_act = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)
        x = self.seg_bias(_flatten_channel_as_conv(x_main))
        z_main = self.encoder(x)

        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            z_aux = self.aux_to_lat(self.aux_enc(x_aux).reshape(b, -1))
            z = self.fusion(torch.cat([z_main, z_aux], dim=1))
        else:
            z = z_main

        # decoder output: [B, profile_len] → [B, 1, profile_len]
        y_prof = self.out_act(self.decoder(z).unsqueeze(1))
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 3. LSTM AE
# =========================================================
class LSTMAE(nn.Module):
    """
    LSTM encoder/decoder with early-fusion aux (concatenated at each time step).
    x_main [B,W,128] + x_aux [B,W,2] → concat → [B,W,130] → LSTM.
    The final hidden state of the encoder is used as the latent representation.
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.profile_len = profile_len
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.num_layers  = num_layers
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        input_dim = profile_len * in_channels + aux_dim

        # encoder LSTM: sequence of window cycles
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # decoder: latent → profile
        # repeat latent as seed, LSTM decodes to profile_len steps
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_proj = nn.Linear(latent_dim, 1)  # one voltage value per time step
        self.out_act = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)

        # Apply seg_bias on the profile dimension
        x_seg = _flatten_channel_as_token(self.seg_bias(x_main))

        x = _cat_aux(x_seg, x_aux)

        # Encoder: use the final hidden state as latent
        _, (h_n, _) = self.encoder_lstm(x)
        z = h_n[-1]  # [B, latent_dim] from the top LSTM layer

        # Decoder: tile latent profile_len times and feed into the LSTM
        dec_input = z.unsqueeze(1).expand(-1, self.profile_len, -1)  # [B, 128, latent_dim]
        dec_out, _ = self.decoder_lstm(dec_input)                    # [B, 128, latent_dim]
        y_prof = self.out_act(self.decoder_proj(dec_out).permute(0, 2, 1))  # [B, 1, 128]

        y_cap = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 4. Bidirectional LSTM AE
# =========================================================
class BiLSTMAE(nn.Module):
    """
    Bidirectional LSTM encoder + late fusion aux
    Bidirectional makes sense because the window is centered on the target cycle.
    BiLSTM output → forward/backward concat → linear → latent
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.profile_len = profile_len
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.num_layers  = num_layers
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        input_dim = profile_len * in_channels + aux_dim

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM: forward + backward → latent_dim * 2
        self.to_latent = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.ReLU(),
        )

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_proj = nn.Linear(latent_dim, 1)
        self.out_act = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)

        x_seg = _flatten_channel_as_token(self.seg_bias(x_main))
        x = _cat_aux(x_seg, x_aux)

        _, (h_n, _) = self.encoder_lstm(x)
        # h_n: [num_layers*2, B, latent_dim]
        # Concatenate forward and backward final hidden states from the top layer
        h_fwd = h_n[-2]  # forward
        h_bwd = h_n[-1]  # backward
        z = self.to_latent(torch.cat([h_fwd, h_bwd], dim=-1))  # [B, latent_dim]

        dec_input = z.unsqueeze(1).expand(-1, self.profile_len, -1)
        dec_out, _ = self.decoder_lstm(dec_input)
        y_prof = self.out_act(self.decoder_proj(dec_out).permute(0, 2, 1))

        y_cap = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 5. Transformer AE
# =========================================================
class TransformerAE(nn.Module):
    """
    Transformer encoder + late fusion aux as condition token
    Each cycle becomes a token. [CLS] + window tokens go through a Transformer
    encoder, and the [CLS] output is used as the latent.
    Aux is encoded separately and added to each token embedding.
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.profile_len = profile_len
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        # profile → token embedding
        self.input_proj = nn.Linear(profile_len * in_channels, latent_dim)

        # aux → token-level embedding (one per cycle)
        self.aux_proj = nn.Linear(aux_dim, latent_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        # positional encoding (window+1 for CLS)
        self.pos_enc = nn.Parameter(torch.zeros(1, window_size + 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # decoder: latent → profile via MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, profile_len),
        )

        self.out_act = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)

        x_seg = _flatten_channel_as_token(self.seg_bias(x_main))

        # token embedding: [B, W, latent_dim]
        tokens = self.input_proj(x_seg)
        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            tokens = tokens + self.aux_proj(x_aux)

        # CLS token prepend
        cls = self.cls_token.expand(b, -1, -1)        # [B, 1, latent_dim]
        tokens = torch.cat([cls, tokens], dim=1)      # [B, W+1, latent_dim]
        tokens = tokens + self.pos_enc[:, :tokens.size(1), :]

        # transformer encoding
        out = self.transformer(tokens)                # [B, W+1, latent_dim]
        z = out[:, 0, :]                              # CLS token → latent [B, latent_dim]

        y_prof = self.out_act(self.decoder(z).unsqueeze(1))  # [B, 1, profile_len]
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 6. 1D UNet AE
# =========================================================
class UNet1DAE(nn.Module):
    """
    1D UNet with skip connections + late fusion aux
    Encoder downsamples 128 → 64 → 32 → 16 with skip connections.
    Decoder: 16→32→64→128 (skip concat)
    Useful for recovering fine-grained profile shape.
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        aux_embed_dim: int = 8,
        base_ch: int = 32,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.total_in_ch = window_size * in_channels
        self.latent_dim  = latent_dim
        self.base_ch     = base_ch
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        # Encoder blocks (each block: Conv → BN → ReLU → MaxPool)
        def enc_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
            )

        self.enc1 = enc_block(self.total_in_ch, base_ch)  # [B, 32, 128]
        self.enc2 = enc_block(base_ch, base_ch * 2)       # [B, 64, 64]
        self.enc3 = enc_block(base_ch * 2, base_ch * 4)   # [B, 128, 32]
        self.pool = nn.MaxPool1d(2)

        # bottleneck
        self.bottleneck = enc_block(base_ch * 4, base_ch * 8)  # [B, 256, 16]
        enc_len = _conv_flat_len(profile_len)
        bottleneck_feat = (base_ch * 8) * enc_len
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(bottleneck_feat, latent_dim), nn.ReLU(),
        )

        # aux late fusion
        self.aux_enc, self.aux_to_lat = _aux_late_fusion(aux_dim, window_size, aux_embed_dim, latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim), nn.ReLU(),
        )

        # Decoder blocks (skip concat doubles in_channels)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_feat), nn.ReLU(),
        )

        def dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
            )

        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = dec_block(base_ch * 8, base_ch * 4)  # skip concat → *8

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = dec_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = dec_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv1d(base_ch, 1, 1)
        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)
        x = self.seg_bias(_flatten_channel_as_conv(x_main))
        # x: [B, W, 128] — treat W as input channels, profile_len as spatial dim

        s1 = self.enc1(x)              # [B, 32, 128]
        s2 = self.enc2(self.pool(s1))  # [B, 64,  64]
        s3 = self.enc3(self.pool(s2))  # [B, 128, 32]
        bn = self.bottleneck(self.pool(s3))  # [B, 256, 16]

        z_main = self.to_latent(bn)

        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            z_aux = self.aux_to_lat(self.aux_enc(x_aux).reshape(b, -1))
            z = self.fusion(torch.cat([z_main, z_aux], dim=1))
        else:
            z = z_main

        # decoder
        h = self.from_latent(z).view(b, self.base_ch * 8, _conv_flat_len(x_main.shape[-1]))

        h = self.dec3(torch.cat([self.up3(h), s3], dim=1))  # [B, 128, 32]
        h = self.dec2(torch.cat([self.up2(h), s2], dim=1))  # [B, 64,  64]
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1))  # [B, 32, 128]

        y_prof = self.out_act(self.out_conv(h))  # [B, 1, 128]
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# 7. Conv + LSTM Hybrid AE
# =========================================================
class ConvLSTMAE(nn.Module):
    """
    Conv1d encodes each cycle's profile into an embedding, and an LSTM
    processes the resulting window sequence.
      Conv : local profile-shape features
      LSTM : cycle-to-cycle trend
      aux  : early fusion at the LSTM input
    """
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        conv_embed_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size   = window_size
        self.profile_len   = profile_len
        self.in_channels   = in_channels
        self.latent_dim    = latent_dim
        self.conv_embed_dim = conv_embed_dim
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        # Conv: per-cycle profile [128] → embedding [conv_embed_dim]
        # Process one cycle at a time: apply Conv1d on shape (batch*window, 1, 128)
        self.cycle_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 128→64
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(), # 64→32
            nn.Conv1d(32, conv_embed_dim, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 32→16
            nn.AdaptiveAvgPool1d(1),  # → conv_embed_dim
            nn.Flatten(),
        )

        # LSTM: [B, W, conv_embed_dim + aux_dim] → sequence encoding
        lstm_input_dim = conv_embed_dim + aux_dim
        self.encoder_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # decoder: latent → profile
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, profile_len),
        )

        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        if x_main.dim() == 4:
            b, w, c, l = x_main.shape
            x_seg = self.seg_bias(x_main)
            x_flat = x_seg.reshape(b * w, c, l)
        else:
            b, w, l = x_main.shape
            x_seg = self.seg_bias(x_main)
            x_flat = x_seg.reshape(b * w, 1, l)

        # Embed each cycle's profile with the Conv encoder
        # [B, W, 128] → [B*W, 1, 128] → conv → [B*W, conv_embed_dim] → [B, W, conv_embed_dim]
        cycle_embed = self.cycle_conv(x_flat).view(b, w, self.conv_embed_dim)

        # early fusion: conv embedding + aux
        lstm_input = _cat_aux(cycle_embed, x_aux)

        _, (h_n, _) = self.encoder_lstm(lstm_input)
        z = h_n[-1]  # [B, latent_dim]

        y_prof = self.out_act(self.decoder(z).unsqueeze(1))  # [B, 1, 128]
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# Factory
# =========================================================
MODEL_REGISTRY = {
    "conv":        ConvAE,
    "mlp":         MLPAE,
    "lstm":        LSTMAE,
    "bilstm":      BiLSTMAE,
    "transformer": TransformerAE,
    "unet":        UNet1DAE,
    "conv_lstm":   ConvLSTMAE,
}


def build_model(
    model_type: str,
    profile_len: int = 128,
    window_size: int = 64,
    in_channels: int = 1,
    aux_dim: int = 2,
    latent_dim: int = 128,
    out_activation: str = "sigmoid",
    use_segment_bias: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Parameters
    ----------
    model_type : str
        'conv' | 'mlp' | 'lstm' | 'bilstm' | 'transformer' | 'unet' | 'conv_lstm'
    **kwargs : architecture-specific hyperparameters
        conv/unet    : aux_embed_dim, base_ch
        mlp          : hidden_dim, aux_embed_dim
        lstm/bilstm  : num_layers, dropout
        transformer  : nhead, num_encoder_layers, dim_feedforward, dropout
        conv_lstm    : conv_embed_dim, num_layers, dropout

    Returns
    -------
    nn.Module with forward(x_main, x_aux) -> (y_prof, y_cap, z)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from {list(MODEL_REGISTRY.keys())}")

    cls = MODEL_REGISTRY[model_type]
    return cls(
        profile_len=profile_len,
        window_size=window_size,
        in_channels=in_channels,
        aux_dim=aux_dim,
        latent_dim=latent_dim,
        out_activation=out_activation,
        use_segment_bias=use_segment_bias,
        **kwargs,
    )


# =========================================================
# Parameter count utility
# =========================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    profile_len: int = 128,
    window_size: int = 64,
    aux_dim: int = 2,
    latent_dim: int = 128,
):
    """Print parameter counts for every registered architecture."""
    print(f"{'Model':<15} {'Params':>12}")
    print("-" * 30)
    for name in MODEL_REGISTRY:
        try:
            m = build_model(
                name,
                profile_len=profile_len,
                window_size=window_size,
                aux_dim=aux_dim,
                latent_dim=latent_dim,
            )
            n = count_parameters(m)
            print(f"{name:<15} {n:>12,}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")


if __name__ == "__main__":
    # sanity check
    B, W, L, A = 4, 64, 128, 2
    x_main = torch.randn(B, W, L)
    x_aux  = torch.randn(B, W, A)

    print("=== Sanity Check ===")
    for name in MODEL_REGISTRY:
        model = build_model(name, profile_len=L, window_size=W, aux_dim=A)
        model.eval()
        with torch.no_grad():
            y_prof, y_cap, z = model(x_main, x_aux)
        assert y_prof.shape == (B, 1, L), f"{name}: y_prof shape {y_prof.shape}"
        assert y_cap.shape  == (B, 2),    f"{name}: y_cap shape {y_cap.shape}"
        assert z.shape      == (B, 128),  f"{name}: z shape {z.shape}"
        print(f"  {name:<15} OK  | params={count_parameters(model):,}")

    print()
    print_model_summary()

"""
generative_models.py
====================
Generative model architectures for BatteryOCV.

Models:
    "conv_vae"        Conv1d VAE
    "lstm_vae"        LSTM VAE
    "transformer_vae" Transformer VAE
    "conv_gan"        Conv1d GAN (Generator + Discriminator)

Common interface (same as AE):
    forward(x_main, x_aux) -> (y_prof, y_cap, z)
    x_main : [B, W, 128]
    x_aux  : [B, W, aux_dim]
    y_prof : [B, 1, 128]
    y_cap  : [B, 2]
    z      : [B, latent_dim]

Inference:
    VAE  : z = mu (sigma ignored, deterministic)
    GAN  : Generator only, discriminator discarded

Training-time additional outputs:
    VAE  : forward(..., training=True) -> (y_prof, y_cap, z, mu, logvar)
    GAN  : discriminator called separately as disc(y_prof) -> [B, 1]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models import (
    SegmentBias, _cap_head, _out_act, _aux_late_fusion,
    _flatten_channel_as_conv, _flatten_channel_as_token, _cat_aux,
    count_parameters,
)
from pipeline import move_batch_to_device


# =========================================================
# Shared Conv-based encoder / decoder blocks
# =========================================================
def _conv_encoder(window_size: int, in_channels: int = 1, enc_dim_ch: int = 16) -> Tuple[nn.Module, int]:
    encoder = nn.Sequential(
        nn.Conv1d(window_size * in_channels, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv1d(64, 32,       kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv1d(32, enc_dim_ch, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    )
    flatten_dim = enc_dim_ch * 16
    return encoder, flatten_dim


def _conv_decoder(latent_dim: int, enc_dim_ch: int = 16) -> Tuple[nn.Module, nn.Module]:
    from_latent = nn.Sequential(
        nn.Linear(latent_dim, enc_dim_ch * 16), nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.ConvTranspose1d(enc_dim_ch, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose1d(64,  1, 3, stride=2, padding=1, output_padding=1),
    )
    return from_latent, decoder


# =========================================================
# 1. Conv VAE
# =========================================================
class ConvVAE(nn.Module):
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
        self.latent_dim  = latent_dim
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        self.encoder, enc_flat = _conv_encoder(window_size, in_channels)
        self.aux_enc, self.aux_to_lat = _aux_late_fusion(aux_dim, window_size, aux_embed_dim, latent_dim)
        fused_dim = enc_flat + latent_dim // 2

        self.fc_mu     = nn.Linear(fused_dim, latent_dim)
        self.fc_logvar = nn.Linear(fused_dim, latent_dim)

        self.from_latent, self.decoder = _conv_decoder(latent_dim)
        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def encode(self, x_main, x_aux):
        b = x_main.size(0)
        x = self.seg_bias(_flatten_channel_as_conv(x_main))
        h = self.encoder(x).flatten(1)
        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            z_aux = self.aux_to_lat(self.aux_enc(x_aux).reshape(b, -1))
            fused = torch.cat([h, z_aux], dim=1)
        else:
            fused = torch.cat([h, torch.zeros(b, self.latent_dim // 2, device=h.device)], dim=1)
        return self.fc_mu(fused), self.fc_logvar(fused)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar.clamp(-10, 10))
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        b = z.size(0)
        h = self.from_latent(z).view(b, 16, 16)
        return self.out_act(self.decoder(h))

    def forward(self, x_main, x_aux):
        mu, logvar = self.encode(x_main, x_aux)
        z = self.reparameterize(mu, logvar)
        y_prof = self.decode(z)
        y_cap  = self.cap_head(mu)
        if self.training:
            return y_prof, y_cap, z, mu, logvar
        return y_prof, y_cap, z


# =========================================================
# 2. LSTM VAE
# =========================================================
class LSTMVAE(nn.Module):
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
            input_size=input_dim, hidden_size=latent_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu     = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim, hidden_size=latent_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_proj = nn.Linear(latent_dim, 1)
        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def encode(self, x_main, x_aux):
        x_seg = _flatten_channel_as_token(self.seg_bias(x_main))
        x = _cat_aux(x_seg, x_aux)
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar.clamp(-10, 10)) * torch.randn_like(mu)
        return mu

    def decode(self, z):
        dec_input = z.unsqueeze(1).expand(-1, self.profile_len, -1)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.out_act(self.decoder_proj(dec_out).permute(0, 2, 1))

    def forward(self, x_main, x_aux):
        mu, logvar = self.encode(x_main, x_aux)
        z = self.reparameterize(mu, logvar)
        y_prof = self.decode(z)
        y_cap  = self.cap_head(mu)
        if self.training:
            return y_prof, y_cap, z, mu, logvar
        return y_prof, y_cap, z


# =========================================================
# 3. Transformer VAE
# =========================================================
class TransformerVAE(nn.Module):
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

        self.input_proj = nn.Linear(profile_len * in_channels, latent_dim)
        self.aux_proj   = nn.Linear(aux_dim, latent_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos_enc    = nn.Parameter(torch.zeros(1, window_size + 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_mu     = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, profile_len),
        )
        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def encode(self, x_main, x_aux):
        b = x_main.size(0)
        x_seg = _flatten_channel_as_token(self.seg_bias(x_main))
        tokens = self.input_proj(x_seg)
        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            tokens = tokens + self.aux_proj(x_aux)
        cls    = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_enc[:, :tokens.size(1), :]
        out = self.transformer(tokens)
        h   = out[:, 0, :]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar.clamp(-10, 10)) * torch.randn_like(mu)
        return mu

    def decode(self, z):
        return self.out_act(self.decoder(z).unsqueeze(1))

    def forward(self, x_main, x_aux):
        mu, logvar = self.encode(x_main, x_aux)
        z = self.reparameterize(mu, logvar)
        y_prof = self.decode(z)
        y_cap  = self.cap_head(mu)
        if self.training:
            return y_prof, y_cap, z, mu, logvar
        return y_prof, y_cap, z


# =========================================================
# 4. Conv GAN
# =========================================================
class ConvGenerator(nn.Module):
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
        self.latent_dim = latent_dim
        self.seg_bias = SegmentBias(profile_len, profile_len // 2) if use_segment_bias else nn.Identity()

        self.encoder, enc_flat = _conv_encoder(window_size, in_channels)
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(enc_flat, latent_dim), nn.ReLU())
        self.aux_enc, self.aux_to_lat = _aux_late_fusion(aux_dim, window_size, aux_embed_dim, latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim), nn.ReLU(),
        )
        self.from_latent, self.decoder = _conv_decoder(latent_dim)
        self.out_act  = _out_act(out_activation)
        self.cap_head = _cap_head(latent_dim)

    def forward(self, x_main, x_aux):
        b = x_main.size(0)
        x = _flatten_channel_as_conv(self.seg_bias(x_main))
        z_main = self.to_latent(self.encoder(x))
        if x_aux is not None and x_aux.numel() > 0 and x_aux.shape[-1] > 0:
            z_aux = self.aux_to_lat(self.aux_enc(x_aux).reshape(b, -1))
        else:
            z_aux = torch.zeros(b, self.latent_dim // 2, device=z_main.device)
        z = self.fusion(torch.cat([z_main, z_aux], dim=1))
        y_prof = self.out_act(self.decoder(self.from_latent(z).view(b, 16, 16)))
        y_cap  = self.cap_head(z)
        return y_prof, y_cap, z


class ConvDiscriminator(nn.Module):
    def __init__(self, profile_len: int = 128, base_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base_ch,      kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch, base_ch*2,   kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*2), nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*4), nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch*4, base_ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*8), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(base_ch * 8 * 8, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConvGAN(nn.Module):
    def __init__(
        self,
        profile_len: int = 128,
        window_size: int = 64,
        in_channels: int = 1,
        aux_dim: int = 2,
        latent_dim: int = 128,
        aux_embed_dim: int = 8,
        disc_base_ch: int = 16,
        out_activation: str = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.generator = ConvGenerator(
            profile_len=profile_len, window_size=window_size,
            in_channels=in_channels, aux_dim=aux_dim, latent_dim=latent_dim,
            aux_embed_dim=aux_embed_dim,
            out_activation=out_activation,
            use_segment_bias=use_segment_bias,
        )
        self.discriminator = ConvDiscriminator(profile_len=profile_len, base_ch=disc_base_ch)

    def forward(self, x_main, x_aux):
        return self.generator(x_main, x_aux)

    def discriminate(self, y_prof):
        return self.discriminator(y_prof)

    def generator_params(self):
        return self.generator.parameters()

    def discriminator_params(self):
        return self.discriminator.parameters()


# =========================================================
# VAE Loss
# =========================================================
class VAELoss(nn.Module):
    def __init__(
        self,
        prof_weight: float = 100.0,
        cap_weight:  float = 5.0,
        diff_weight: float = 0.0,
        beta: float = 1.0,
        use_diff_loss: bool = False,
        cap_loss_type: str = "smoothl1",
    ):
        super().__init__()
        self.prof_weight   = prof_weight
        self.cap_weight    = cap_weight
        self.diff_weight   = diff_weight
        self.beta          = beta
        self.use_diff_loss = use_diff_loss
        self.prof_crit = nn.MSELoss()
        self.cap_crit  = nn.SmoothL1Loss() if cap_loss_type == "smoothl1" else nn.MSELoss()

    def kl_loss(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, yprof_pred, yprof_true, ycap_pred, ycap_true,
                mu, logvar, beta: Optional[float] = None):
        beta = beta if beta is not None else self.beta
        loss_prof = self.prof_crit(yprof_pred, yprof_true)
        loss_cap  = self.cap_crit(ycap_pred, ycap_true)
        loss_kl   = self.kl_loss(mu, logvar)
        loss_diff = torch.zeros((), device=yprof_pred.device)
        if self.use_diff_loss:
            loss_diff = self.prof_crit(
                torch.diff(yprof_pred, dim=-1),
                torch.diff(yprof_true, dim=-1),
            )
        total = (self.prof_weight * loss_prof + self.diff_weight * loss_diff
                 + self.cap_weight * loss_cap + beta * loss_kl)
        return total, {
            "loss_total":       total.item(),
            "loss_prof":        loss_prof.item(),
            "loss_diff":        loss_diff.item(),
            "loss_cap":         loss_cap.item(),
            "loss_kl":          loss_kl.item(),
            "loss_prof_scaled": self.prof_weight * loss_prof.item(),
            "loss_cap_scaled":  self.cap_weight  * loss_cap.item(),
            "loss_kl_scaled":   beta             * loss_kl.item(),
        }


# =========================================================
# GAN Loss
# =========================================================
class GANLoss(nn.Module):
    def __init__(
        self,
        prof_weight: float = 100.0,
        cap_weight:  float = 5.0,
        adv_weight:  float = 1.0,
        diff_weight: float = 0.0,
        use_diff_loss: bool = False,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.prof_weight    = prof_weight
        self.cap_weight     = cap_weight
        self.adv_weight     = adv_weight
        self.diff_weight    = diff_weight
        self.use_diff_loss  = use_diff_loss
        self.label_smoothing = label_smoothing
        self.prof_crit = nn.MSELoss()
        self.cap_crit  = nn.SmoothL1Loss()
        self.adv_crit  = nn.BCEWithLogitsLoss()

    def generator_loss(self, fake_prof, real_prof, fake_cap, real_cap, fake_score):
        b = fake_score.size(0)
        real_labels = torch.ones(b, 1, device=fake_score.device)
        loss_adv  = self.adv_crit(fake_score, real_labels)
        loss_prof = self.prof_crit(fake_prof, real_prof)
        loss_cap  = self.cap_crit(fake_cap, real_cap)
        loss_diff = torch.zeros((), device=fake_prof.device)
        if self.use_diff_loss:
            loss_diff = self.prof_crit(
                torch.diff(fake_prof, dim=-1), torch.diff(real_prof, dim=-1))
        total_G = (self.adv_weight * loss_adv + self.prof_weight * loss_prof
                   + self.diff_weight * loss_diff + self.cap_weight * loss_cap)
        return total_G, {
            "loss_G_total":     total_G.item(),
            "loss_adv":         loss_adv.item(),
            "loss_prof":        loss_prof.item(),
            "loss_cap":         loss_cap.item(),
            "loss_diff":        loss_diff.item(),
            "loss_prof_scaled": self.prof_weight * loss_prof.item(),
            "loss_cap_scaled":  self.cap_weight  * loss_cap.item(),
        }

    def discriminator_loss(self, real_score, fake_score):
        b = real_score.size(0)
        real_labels = torch.full((b, 1), 1.0 - self.label_smoothing, device=real_score.device)
        fake_labels = torch.zeros(b, 1, device=fake_score.device)
        loss_real = self.adv_crit(real_score, real_labels)
        loss_fake = self.adv_crit(fake_score, fake_labels)
        total_D   = (loss_real + loss_fake) * 0.5
        return total_D, {
            "loss_D_total": total_D.item(),
            "loss_D_real":  loss_real.item(),
            "loss_D_fake":  loss_fake.item(),
        }


# =========================================================
# Training-loop helpers
# =========================================================
def run_epoch_vae(
    model: nn.Module,
    loader,
    loss_fn: VAELoss,
    device: torch.device,
    optimizer=None,
    beta: float = 1.0,
    grad_clip: float = 1.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = {k: 0.0 for k in [
        "loss_total", "loss_prof", "loss_cap", "loss_kl",
        "loss_prof_scaled", "loss_cap_scaled", "loss_kl_scaled"
    ]}
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
            bs = x_main.size(0)

            out = model(x_main, x_aux)
            if len(out) == 5:
                y_prof_pred, y_cap_pred, z, mu, logvar = out
            else:
                y_prof_pred, y_cap_pred, z = out
                mu, logvar = z, torch.zeros_like(z)

            loss, loss_dict = loss_fn(
                y_prof_pred, y_prof, y_cap_pred, y_cap, mu, logvar, beta=beta)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for k in total:
                if k in loss_dict:
                    total[k] += loss_dict[k] * bs
            n += bs

    return {k: v / max(n, 1) for k, v in total.items()}


def run_epoch_gan(
    model: ConvGAN,
    loader,
    loss_fn: GANLoss,
    device: torch.device,
    optimizer_G=None,
    optimizer_D=None,
    grad_clip: float = 1.0,
    n_disc_steps: int = 1,
):
    is_train = (optimizer_G is not None) and (optimizer_D is not None)
    model.train() if is_train else model.eval()

    total_G = {"loss_G_total": 0.0, "loss_adv": 0.0,
               "loss_prof": 0.0, "loss_cap": 0.0}
    total_D = {"loss_D_total": 0.0, "loss_D_real": 0.0, "loss_D_fake": 0.0}
    total_prof_rmse = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
            bs = x_main.size(0)

            if is_train:
                for _ in range(n_disc_steps):
                    with torch.no_grad():
                        fake_prof, _, _ = model(x_main, x_aux)
                    real_score = model.discriminate(y_prof)
                    fake_score = model.discriminate(fake_prof.detach())
                    loss_D, dict_D = loss_fn.discriminator_loss(real_score, fake_score)
                    optimizer_D.zero_grad(set_to_none=True)
                    loss_D.backward()
                    torch.nn.utils.clip_grad_norm_(model.discriminator_params(), grad_clip)
                    optimizer_D.step()

                fake_prof, fake_cap, _ = model(x_main, x_aux)
                fake_score = model.discriminate(fake_prof)
                loss_G, dict_G = loss_fn.generator_loss(
                    fake_prof, y_prof, fake_cap, y_cap, fake_score)
                optimizer_G.zero_grad(set_to_none=True)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(model.generator_params(), grad_clip)
                optimizer_G.step()
            else:
                fake_prof, fake_cap, _ = model(x_main, x_aux)
                fake_score = model.discriminate(fake_prof)
                _, dict_G = loss_fn.generator_loss(
                    fake_prof, y_prof, fake_cap, y_cap, fake_score)
                _, dict_D = loss_fn.discriminator_loss(
                    model.discriminate(y_prof),
                    model.discriminate(fake_prof.detach()),
                )

            for k in total_G:
                if k in dict_G: total_G[k] += dict_G[k] * bs
            for k in total_D:
                if k in dict_D: total_D[k] += dict_D[k] * bs
            total_prof_rmse += torch.sqrt(
                ((fake_prof - y_prof) ** 2).mean()).item() * bs
            n += bs

    result = {k: v / max(n, 1) for k, v in {**total_G, **total_D}.items()}
    result["val_prof_rmse"] = total_prof_rmse / max(n, 1)
    result["loss_prof_scaled"] = total_G.get("loss_prof", 0.0) / max(n, 1) * loss_fn.prof_weight
    return result


# =========================================================
# Factory
# =========================================================
GENERATIVE_MODEL_REGISTRY = {
    "conv_vae":        ConvVAE,
    "lstm_vae":        LSTMVAE,
    "transformer_vae": TransformerVAE,
    "conv_gan":        ConvGAN,
}


def build_generative_model(
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
    if model_type not in GENERATIVE_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from {list(GENERATIVE_MODEL_REGISTRY.keys())}"
        )
    cls = GENERATIVE_MODEL_REGISTRY[model_type]
    base_kwargs = dict(
        profile_len=profile_len,
        window_size=window_size,
        aux_dim=aux_dim,
        latent_dim=latent_dim,
        out_activation=out_activation,
        use_segment_bias=use_segment_bias,
        **kwargs,
    )
    if model_type in {"conv_vae", "lstm_vae", "transformer_vae", "conv_gan"}:
        base_kwargs["in_channels"] = in_channels
    return cls(**base_kwargs)

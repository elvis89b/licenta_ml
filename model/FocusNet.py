import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_module import FocusAttention, ChannelAttention, SpatialAttention
from .layers import BasicConv2d, DeformableConv2d, eca_layer
from .pvtv2 import pvt_v2_b4


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean"
        )

        return bce + dice_loss


class WeightedDiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, pixel_weight=None):
        if pixel_weight is None:
            pixel_weight = torch.ones_like(targets)

        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        bce = (bce * pixel_weight).sum() / (pixel_weight.sum() + 1e-6)

        dims = (1, 2, 3)

        intersection = (pixel_weight * probs * targets).sum(dim=dims)

        denominator = (
            (pixel_weight * probs).sum(dim=dims)
            + (pixel_weight * targets).sum(dim=dims)
        )

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        return bce + dice_loss.mean()


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()

        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DEM(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DEM, self).__init__()

        self.down_scale = 1

        if out_channels is None:
            out_channels = in_channels

        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels

        self.aspp1 = nn.Sequential(
            _ASPPModuleDeformable(
                in_channels // 4,
                self.in_channelster,
                (1, 3),
                padding=0
            ),
            _ASPPModuleDeformable(
                self.in_channelster,
                self.in_channelster,
                (3, 1),
                padding=0
            )
        )

        self.aspp2 = nn.Sequential(
            _ASPPModuleDeformable(
                in_channels // 4,
                self.in_channelster,
                (1, 3),
                padding=0
            ),
            _ASPPModuleDeformable(
                self.in_channelster,
                self.in_channelster,
                (3, 1),
                padding=0
            )
        )

        self.aspp3 = nn.Sequential(
            _ASPPModuleDeformable(
                in_channels // 4,
                self.in_channelster,
                (1, 3),
                padding=0
            ),
            _ASPPModuleDeformable(
                self.in_channelster,
                self.in_channelster,
                (3, 1),
                padding=0
            )
        )

        self.aspp4 = nn.Sequential(
            _ASPPModuleDeformable(
                in_channels // 4,
                self.in_channelster,
                (1, 3),
                padding=0
            ),
            _ASPPModuleDeformable(
                self.in_channelster,
                self.in_channelster,
                (3, 1),
                padding=0
            )
        )

        self.eca = eca_layer(self.in_channelster)

        self.conv1 = nn.Conv2d(
            self.in_channelster * 4,
            self.in_channelster,
            1,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            self.in_channelster,
            out_channels,
            1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.ret = lambda x, target: F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        x_1, x_2, x_3, x_4 = torch.split(
            x,
            self.in_channels // 4,
            dim=1
        )

        x1 = self.aspp1(x_1)
        x2 = self.aspp2(x_2)
        x3 = self.aspp3(x_3)
        x4 = self.aspp4(x_4)

        x2 = self.ret(x2, x1)
        x3 = self.ret(x3, x1)
        x4 = self.ret(x4, x1)

        x_ = torch.cat((x1, x2, x3, x4), dim=1)

        x_ = self.conv1(x_)

        x = self.eca(x_)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class CIDM_M(nn.Module):
    def __init__(self, channel):
        super(CIDM_M, self).__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.up03 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.up04 = nn.Upsample(
            scale_factor=4,
            mode="bilinear",
            align_corners=True
        )

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv3 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

        self.out_conv = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        xh = x1

        xm = self.conv_upsample1(self.upsample(x1)) * x2

        xl = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        xm = self.up03(xm)
        xh = self.up04(xh)

        xm_ca = self.ca(xh) * xm
        xm_ca = self.conv1(xm_ca)

        xm_sa = self.sa(xl) * xm
        xm_sa = self.conv2(xm_sa)

        x = self.conv3(torch.cat((xl, xm_ca, xm_sa, xh), dim=1))
        out = self.out_conv(x)

        return x, out


class CIDM_A(nn.Module):
    def __init__(self, channel):
        super(CIDM_A, self).__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.up03 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.up04 = nn.Upsample(
            scale_factor=4,
            mode="bilinear",
            align_corners=True
        )

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv3 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

        self.out_conv = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        xh = x1

        xm = self.conv_upsample1(self.upsample(x1)) + x2

        xl = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            + self.conv_upsample3(self.upsample(x2))
            + x3
        )

        xm = self.up03(xm)
        xh = self.up04(xh)

        xm_ca = self.ca(xh) * xm
        xm_ca = self.conv1(xm_ca)

        xm_sa = self.sa(xl) * xm
        xm_sa = self.conv2(xm_sa)

        x = self.conv3(torch.cat((xl, xm_ca, xm_sa, xh), dim=1))
        out = self.out_conv(x)

        return x, out


class FocusNet(nn.Module):
    def __init__(self, channel=32):
        super(FocusNet, self).__init__()

        self.pvt = pvt_v2_b4()

        path = "pretrained_pth/pvt_v2_b4.pth"

        save_model = torch.load(path, map_location="cpu")
        model_dict = self.pvt.state_dict()

        state_dict = {
            k: v
            for k, v in save_model.items()
            if k in model_dict.keys()
        }

        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)

        self.Translayer_pvt2 = BasicConv2d(128, channel, 1)
        self.Translayer_pvt3 = BasicConv2d(320, channel, 1)
        self.Translayer_pvt4 = BasicConv2d(512, channel, 1)

        self.context = DEM(64, channel)

        self.decoder1 = CIDM_M(channel)
        self.decoder2 = CIDM_A(channel)

        self.attention = FocusAttention(channel, channel)

        self.seg_loss = DiceBCELoss()
        self.weighted_seg_loss = WeightedDiceBCELoss()

        self.loss_w1 = 0.25
        self.loss_w2 = 0.25
        self.loss_w3 = 0.50
        self.loss_w4 = 0.50

        self.final_loss_weight = 1.00
        self.band_loss_weight = 0.20
        self.dynamic_expert_weight = 0.25
        self.decoder_consistency_weight = 0.05
        self.router_entropy_weight = 0.005

        self.ugel_expert_scale = 1.00
        self.spectral_expert_scale = 0.75
        self.region_expert_scale = 0.75

        self.band_kernel_size = 7
        self.safe_region_kernel_size = 9

        self.band_head = nn.Sequential(
            BasicConv2d(channel * 2, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

        self.router_feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, 16),
            nn.ReLU(inplace=True)
        )

        self.router_mlp = nn.Sequential(
            nn.Linear(20, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(24, 3)
        )

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32
        ).unsqueeze(0)

        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]],
            dtype=torch.float32
        ).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def res(self, x, size):
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False
        )

    def make_band_target(self, mask, kernel_size=7):
        pad = kernel_size // 2

        dilated = F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

        eroded = -F.max_pool2d(
            -mask,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

        band = dilated - eroded

        return torch.clamp(band, 0.0, 1.0)

    def make_safe_regions(self, mask, kernel_size=9):
        pad = kernel_size // 2

        dilated = F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

        eroded = -F.max_pool2d(
            -mask,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

        safe_foreground = torch.clamp(eroded, 0.0, 1.0)
        safe_background = torch.clamp(1.0 - dilated, 0.0, 1.0)

        return safe_foreground, safe_background

    def normalize_uncertainty(self, uncertainty):
        uncertainty = uncertainty.detach()

        u_min = uncertainty.amin(dim=(2, 3), keepdim=True)
        u_max = uncertainty.amax(dim=(2, 3), keepdim=True)

        uncertainty = (uncertainty - u_min) / (
            u_max - u_min + 1e-6
        )

        return torch.clamp(uncertainty, 0.0, 1.0)

    def sobel_edge_map(self, x):
        grad_x = F.conv2d(
            x,
            self.sobel_x,
            padding=1
        )

        grad_y = F.conv2d(
            x,
            self.sobel_y,
            padding=1
        )

        edge = torch.sqrt(
            grad_x.pow(2) + grad_y.pow(2) + 1e-6
        )

        return edge

    def weighted_dice_bce_per_sample(self, logits, targets, pixel_weight):
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        dims = (1, 2, 3)

        bce = (bce * pixel_weight).sum(dim=dims) / (
            pixel_weight.sum(dim=dims) + 1e-6
        )

        intersection = (pixel_weight * probs * targets).sum(dim=dims)

        denominator = (
            (pixel_weight * probs).sum(dim=dims)
            + (pixel_weight * targets).sum(dim=dims)
        )

        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (
            denominator + 1.0
        )

        return bce + dice_loss

    def uncertainty_gated_edge_loss_per_sample(
        self,
        prediction_logits,
        target_mask,
        band_target,
        uncertainty_map
    ):
        edge_focus = torch.clamp(
            band_target * (1.0 + uncertainty_map),
            min=0.0,
            max=2.0
        )

        edge_weight = 1.0 + edge_focus

        return self.weighted_dice_bce_per_sample(
            prediction_logits,
            target_mask,
            pixel_weight=edge_weight
        )

    def spectral_boundary_consistency_loss_per_sample(
        self,
        prediction_logits,
        target_mask
    ):
        pred_prob = torch.sigmoid(prediction_logits)

        pred_edge = self.sobel_edge_map(pred_prob)
        target_edge = self.sobel_edge_map(target_mask)

        pred_edge = F.interpolate(
            pred_edge,
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        )

        target_edge = F.interpolate(
            target_edge,
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        )

        pred_fft = torch.fft.fft2(
            pred_edge.squeeze(1),
            norm="ortho"
        )

        target_fft = torch.fft.fft2(
            target_edge.squeeze(1),
            norm="ortho"
        )

        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))

        pred_mag = torch.fft.fftshift(pred_mag, dim=(-2, -1))
        target_mag = torch.fft.fftshift(target_mag, dim=(-2, -1))

        pred_mag = pred_mag / (
            pred_mag.mean(dim=(-2, -1), keepdim=True) + 1e-6
        )

        target_mag = target_mag / (
            target_mag.mean(dim=(-2, -1), keepdim=True) + 1e-6
        )

        h, w = pred_mag.shape[-2:]

        yy = torch.linspace(
            -1.0,
            1.0,
            h,
            device=pred_mag.device
        ).view(h, 1)

        xx = torch.linspace(
            -1.0,
            1.0,
            w,
            device=pred_mag.device
        ).view(1, w)

        radius = torch.sqrt(xx.pow(2) + yy.pow(2))
        radius = radius / (radius.max() + 1e-6)

        frequency_weight = 1.0 + radius
        frequency_weight = frequency_weight.unsqueeze(0)

        spectral_difference = torch.abs(
            pred_mag - target_mag
        ) * frequency_weight

        loss = spectral_difference.mean(dim=(-2, -1))

        return loss

    def region_calibration_loss_per_sample(
        self,
        prediction_logits,
        safe_foreground,
        safe_background,
        uncertainty_map
    ):
        probs = torch.sigmoid(prediction_logits)

        confidence = torch.clamp(
            1.0 - uncertainty_map,
            min=0.0,
            max=1.0
        )

        bg_weight = safe_background * confidence
        fg_weight = safe_foreground * confidence

        false_positive_penalty = -torch.log(
            1.0 - probs + 1e-6
        )

        false_negative_penalty = -torch.log(
            probs + 1e-6
        )

        dims = (1, 2, 3)

        background_loss = (
            false_positive_penalty * bg_weight
        ).sum(dim=dims) / (
            bg_weight.sum(dim=dims) + 1e-6
        )

        foreground_loss = (
            false_negative_penalty * fg_weight
        ).sum(dim=dims) / (
            fg_weight.sum(dim=dims) + 1e-6
        )

        region_loss = (
            0.65 * background_loss
            + 0.35 * foreground_loss
        )

        return region_loss

    def selective_decoder_agreement_loss(
        self,
        logit1,
        logit2,
        uncertainty_map
    ):
        p1 = torch.sigmoid(logit1)
        p2 = torch.sigmoid(logit2)

        confidence = torch.clamp(
            1.0 - uncertainty_map,
            min=0.0,
            max=1.0
        )

        return (
            torch.abs(p1 - p2) * confidence
        ).mean()

    def compute_high_low_frequency_score(self, images):
        gray = images.mean(dim=1, keepdim=True)

        gray_small = F.interpolate(
            gray,
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        )

        fft = torch.fft.fft2(
            gray_small.squeeze(1),
            norm="ortho"
        )

        amplitude = torch.abs(fft)
        amplitude = torch.fft.fftshift(
            amplitude,
            dim=(-2, -1)
        )

        h, w = amplitude.shape[-2:]
        center_h = h // 2
        center_w = w // 2
        radius = 8

        low_region = amplitude[
            :,
            center_h - radius:center_h + radius,
            center_w - radius:center_w + radius
        ]

        low_energy = low_region.sum(dim=(-2, -1))
        total_energy = amplitude.sum(dim=(-2, -1))
        high_energy = torch.clamp(
            total_energy - low_energy,
            min=0.0
        )

        ratio = torch.log1p(
            high_energy / (low_energy + 1e-6)
        )

        score = torch.tanh(ratio)

        return score

    def build_router_weights(
        self,
        decoder_feature,
        images,
        prediction_logits,
        uncertainty_full
    ):
        pooled_feature = self.router_feature_projection(
            decoder_feature
        )

        gray = images.mean(dim=1, keepdim=True)
        gray_edge = self.sobel_edge_map(gray)

        edge_energy = gray_edge.mean(dim=(1, 2, 3))
        edge_score = torch.tanh(edge_energy * 3.0)

        uncertainty_score = torch.clamp(
            uncertainty_full.detach().mean(dim=(1, 2, 3)) * 2.0,
            min=0.0,
            max=1.0
        )

        frequency_score = self.compute_high_low_frequency_score(
            images
        ).detach()

        area_ratio = torch.sigmoid(
            prediction_logits.detach()
        ).mean(dim=(1, 2, 3))

        handcrafted_stats = torch.stack(
            [
                uncertainty_score,
                edge_score.detach(),
                frequency_score,
                area_ratio
            ],
            dim=1
        )

        router_input = torch.cat(
            [pooled_feature, handcrafted_stats],
            dim=1
        )

        learned_logits = self.router_mlp(router_input)

        ugel_prior = (
            1.50 * uncertainty_score
            + 0.50 * edge_score.detach()
        )

        spectral_prior = (
            1.00 * frequency_score
            + 0.75 * edge_score.detach()
        )

        region_prior = (
            1.25 * (1.0 - uncertainty_score)
            + 0.75 * (1.0 - edge_score.detach())
        )

        prior_logits = torch.stack(
            [
                ugel_prior,
                spectral_prior,
                region_prior
            ],
            dim=1
        )

        router_logits = prior_logits + 0.35 * learned_logits

        router_weights = F.softmax(
            router_logits,
            dim=1
        )

        return router_weights

    def forward(self, sample):
        x = sample["images"]
        y = sample["masks"]

        base_size = x.shape[-2:]

        pvt = self.pvt(x)

        x1_pvt = pvt[0]
        x2_pvt = pvt[1]
        x3_pvt = pvt[2]
        x4_pvt = pvt[3]

        x2_pvt = self.Translayer_pvt2(x2_pvt)
        x3_pvt = self.Translayer_pvt3(x3_pvt)
        x4_pvt = self.Translayer_pvt4(x4_pvt)

        f1, a1 = self.decoder1(
            x4_pvt,
            x3_pvt,
            x2_pvt
        )

        out1 = self.res(a1, base_size)

        f2, a2 = self.decoder2(
            x4_pvt,
            x3_pvt,
            x2_pvt
        )

        out2 = self.res(a2, base_size)

        x_t = self.context(x1_pvt)

        coarse_uncertainty = torch.abs(
            torch.sigmoid(a1) - torch.sigmoid(a2)
        )

        f3, a3, _ = self.attention(
            f1,
            x_t,
            a1,
            coarse_uncertainty
        )

        out3 = self.res(a3, base_size)

        f4, a4, _ = self.attention(
            f2,
            x_t,
            a2,
            coarse_uncertainty
        )

        out4 = self.res(a4, base_size)

        out = out1 + out2 + out3 + out4

        band_feature = torch.cat([f3, f4], dim=1)
        band_prediction = self.band_head(band_feature)
        band_prediction_up = self.res(
            band_prediction,
            base_size
        )

        decoder_feature = 0.5 * (f3 + f4)

        uncertainty_full = self.res(
            coarse_uncertainty,
            base_size
        )

        uncertainty_full = self.normalize_uncertainty(
            uncertainty_full
        )

        band_target = self.make_band_target(
            y,
            kernel_size=self.band_kernel_size
        )

        safe_foreground, safe_background = self.make_safe_regions(
            y,
            kernel_size=self.safe_region_kernel_size
        )

        final_pixel_weight = (
            1.0
            + 0.70 * band_target
            + 0.45 * uncertainty_full
            + 0.70 * band_target * uncertainty_full
        )

        loss1 = self.seg_loss(out1, y)
        loss2 = self.seg_loss(out2, y)
        loss3 = self.seg_loss(out3, y)
        loss4 = self.seg_loss(out4, y)

        loss_aux = (
            self.loss_w1 * loss1
            + self.loss_w2 * loss2
            + self.loss_w3 * loss3
            + self.loss_w4 * loss4
        )

        loss_final = self.weighted_seg_loss(
            out,
            y,
            pixel_weight=final_pixel_weight
        )

        loss_band = self.weighted_seg_loss(
            band_prediction_up,
            band_target,
            pixel_weight=1.0 + band_target + uncertainty_full
        )

        loss_ugel_vector = self.uncertainty_gated_edge_loss_per_sample(
            prediction_logits=out,
            target_mask=y,
            band_target=band_target,
            uncertainty_map=uncertainty_full
        )

        loss_spectral_vector = self.spectral_boundary_consistency_loss_per_sample(
            prediction_logits=out,
            target_mask=y
        )

        loss_region_vector = self.region_calibration_loss_per_sample(
            prediction_logits=out,
            safe_foreground=safe_foreground,
            safe_background=safe_background,
            uncertainty_map=uncertainty_full
        )

        router_weights = self.build_router_weights(
            decoder_feature=decoder_feature,
            images=x,
            prediction_logits=out,
            uncertainty_full=uncertainty_full
        )

        routed_expert_vector = (
            router_weights[:, 0] * self.ugel_expert_scale * loss_ugel_vector
            + router_weights[:, 1] * self.spectral_expert_scale * loss_spectral_vector
            + router_weights[:, 2] * self.region_expert_scale * loss_region_vector
        )

        loss_dynamic_expert = routed_expert_vector.mean()

        loss_consistency = self.selective_decoder_agreement_loss(
            out1,
            out2,
            uncertainty_full
        )

        router_entropy_loss = (
            router_weights * torch.log(router_weights + 1e-6)
        ).sum(dim=1).mean()

        loss = (
            loss_aux
            + self.final_loss_weight * loss_final
            + self.band_loss_weight * loss_band
            + self.dynamic_expert_weight * loss_dynamic_expert
            + self.decoder_consistency_weight * loss_consistency
            + self.router_entropy_weight * router_entropy_loss
        )

        return {
            "prediction": out,
            "loss": loss,
            "loss_aux": loss_aux.detach(),
            "loss_final": loss_final.detach(),
            "loss_band": loss_band.detach(),
            "loss_dynamic_expert": loss_dynamic_expert.detach(),
            "loss_ugel": loss_ugel_vector.mean().detach(),
            "loss_spectral": loss_spectral_vector.mean().detach(),
            "loss_region": loss_region_vector.mean().detach(),
            "loss_consistency": loss_consistency.detach(),
            "loss_router_entropy": router_entropy_loss.detach(),
            "router_ugel_weight": router_weights[:, 0].mean().detach(),
            "router_spectral_weight": router_weights[:, 1].mean().detach(),
            "router_region_weight": router_weights[:, 2].mean().detach()
        }

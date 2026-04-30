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

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        bce = F.binary_cross_entropy(inputs, targets, reduction="mean")

        return bce + dice_loss


class WeightedDiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, pixel_weight=None):
        if pixel_weight is None:
            pixel_weight = torch.ones_like(targets)

        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (bce * pixel_weight).sum() / (pixel_weight.sum() + 1e-6)

        dims = (1, 2, 3)
        intersection = (pixel_weight * probs * targets).sum(dim=dims)
        denom = (pixel_weight * probs).sum(dim=dims) + (pixel_weight * targets).sum(dim=dims)

        dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice = dice.mean()

        return bce + dice


class PrecisionFocalTverskyLoss(nn.Module):
    """
    Precision-oriented Focal Tversky.

    alpha > beta means false positives are penalized more than false negatives.
    This is intentional because the previous experiment over-segmented strongly.
    """

    def __init__(self, alpha=0.70, beta=0.30, gamma=0.75, smooth=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        dims = (1, 2, 3)

        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = torch.pow(1.0 - tversky, self.gamma)

        return loss.mean()


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

        return self.relu(x)


class DEM(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DEM, self).__init__()

        self.down_scale = 1

        if out_channels is None:
            out_channels = in_channels

        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels

        self.aspp1 = nn.Sequential(
            _ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
            _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0)
        )

        self.aspp2 = nn.Sequential(
            _ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
            _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0)
        )

        self.aspp3 = nn.Sequential(
            _ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
            _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0)
        )

        self.aspp4 = nn.Sequential(
            _ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
            _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0)
        )

        self.eca = eca_layer(self.in_channelster)

        self.conv1 = nn.Conv2d(self.in_channelster * 4, self.in_channelster, 1, bias=False)
        self.conv2 = nn.Conv2d(self.in_channelster, out_channels, 1, bias=False)

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
        x_1, x_2, x_3, x_4 = torch.split(x, self.in_channels // 4, dim=1)

        x1 = self.aspp1(x_1)
        x2 = self.aspp2(x_2)
        x3 = self.aspp3(x_3)
        x4 = self.aspp4(x_4)

        x2 = self.ret(x2, x1)
        x3 = self.ret(x3, x1)
        x4 = self.ret(x4, x1)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_cat = self.conv1(x_cat)

        x = self.eca(x_cat)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class CIDM_M(nn.Module):
    def __init__(self, channel):
        super(CIDM_M, self).__init__()

        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up03 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up04 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

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

        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up03 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up04 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

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
        save_model = torch.load(path)

        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

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
        self.precision_tversky_loss = PrecisionFocalTverskyLoss(
            alpha=0.70,
            beta=0.30,
            gamma=0.75
        )

        self.loss_w1 = 0.35
        self.loss_w2 = 0.35
        self.loss_w3 = 0.65
        self.loss_w4 = 0.65
        self.final_loss_weight = 1.00

        self.precision_tversky_weight = 0.20
        self.consistency_weight = 0.10
        self.band_loss_weight = 0.22

        self.conservative_ugel_weight = 0.14
        self.false_positive_weight = 0.20
        self.background_loss_weight = 0.16
        self.area_loss_weight = 0.04

        self.band_head = nn.Sequential(
            BasicConv2d(channel * 2, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

        self.background_head = nn.Sequential(
            BasicConv2d(channel * 2, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]]
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def res(self, x, size):
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False
        )

    def selective_agreement_loss(self, logit1, logit2, uncertainty_map):
        p1 = torch.sigmoid(logit1)
        p2 = torch.sigmoid(logit2)

        confidence = torch.clamp(
            1.0 - uncertainty_map,
            min=0.0,
            max=1.0
        )

        return (torch.abs(p1 - p2) * confidence).mean()

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

    def dilate_mask(self, mask, kernel_size=13):
        pad = kernel_size // 2

        dilated = F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

        return torch.clamp(dilated, 0.0, 1.0)

    def normalize_uncertainty(self, u):
        u = u.detach()

        u_min = u.amin(dim=(2, 3), keepdim=True)
        u_max = u.amax(dim=(2, 3), keepdim=True)

        u = (u - u_min) / (u_max - u_min + 1e-6)

        return torch.clamp(u, 0.0, 1.0)

    def get_modality_scale(self, sample, scale_type="edge"):
        modalities = sample.get("modalities", None)

        if modalities is None:
            return 1.0

        if isinstance(modalities, str):
            modalities = [modalities]

        edge_scales = {
            "BLI": 1.00,
            "FICE": 0.95,
            "LCI": 0.85,
            "NBI": 1.05,
            "WLI": 0.08,
            "UNKNOWN": 0.75
        }

        fp_scales = {
            "BLI": 1.15,
            "FICE": 1.20,
            "LCI": 1.35,
            "NBI": 1.05,
            "WLI": 0.70,
            "UNKNOWN": 1.00
        }

        bg_scales = {
            "BLI": 1.10,
            "FICE": 1.15,
            "LCI": 1.30,
            "NBI": 1.05,
            "WLI": 0.65,
            "UNKNOWN": 1.00
        }

        if scale_type == "edge":
            scale_map = edge_scales
        elif scale_type == "fp":
            scale_map = fp_scales
        else:
            scale_map = bg_scales

        values = []

        for m in modalities:
            key = str(m).upper()
            values.append(scale_map.get(key, scale_map["UNKNOWN"]))

        return float(sum(values) / len(values))

    def edge_map(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)

        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        return edge

    def conservative_uncertainty_gated_edge_loss(self, logits, mask, sample):
        probs = torch.sigmoid(logits)

        pred_edge = self.edge_map(probs)
        gt_edge = self.edge_map(mask)

        boundary_band = self.make_band_target(mask, kernel_size=7)

        uncertainty = 4.0 * probs * (1.0 - probs)
        uncertainty = uncertainty.detach()
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        edge_error = torch.abs(pred_edge - gt_edge)

        loss = edge_error * boundary_band * (0.25 + 0.75 * uncertainty)
        loss = loss.mean()

        modality_scale = self.get_modality_scale(sample, scale_type="edge")

        return loss * modality_scale

    def false_positive_suppression_loss(self, logits, mask, sample):
        probs = torch.sigmoid(logits)

        dilated_mask = self.dilate_mask(mask, kernel_size=13)

        safe_background = 1.0 - dilated_mask

        high_confidence_fp = probs * probs

        loss = (high_confidence_fp * safe_background).sum() / (
            safe_background.sum() + 1e-6
        )

        modality_scale = self.get_modality_scale(sample, scale_type="fp")

        return loss * modality_scale

    def background_supervision_loss(self, background_logits, mask, sample):
        background_target = 1.0 - mask

        dilated_mask = self.dilate_mask(mask, kernel_size=13)
        safe_background = 1.0 - dilated_mask

        boundary_band = self.make_band_target(mask, kernel_size=9)

        pixel_weight = 1.0 + 1.5 * safe_background + 0.75 * boundary_band

        bce = F.binary_cross_entropy_with_logits(
            background_logits,
            background_target,
            reduction="none"
        )

        bce = (bce * pixel_weight).sum() / (pixel_weight.sum() + 1e-6)

        modality_scale = self.get_modality_scale(sample, scale_type="bg")

        return bce * modality_scale

    def area_regularization_loss(self, logits, mask):
        probs = torch.sigmoid(logits)

        pred_area = probs.mean(dim=(1, 2, 3))
        gt_area = mask.mean(dim=(1, 2, 3))

        return F.smooth_l1_loss(pred_area, gt_area)

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

        f1, a1 = self.decoder1(x4_pvt, x3_pvt, x2_pvt)
        out1 = self.res(a1, base_size)

        f2, a2 = self.decoder2(x4_pvt, x3_pvt, x2_pvt)
        out2 = self.res(a2, base_size)

        x_t = self.context(x1_pvt)

        coarse_uncertainty = torch.abs(torch.sigmoid(a1) - torch.sigmoid(a2))

        f3, a3, _ = self.attention(f1, x_t, a1, coarse_uncertainty)
        out3 = self.res(a3, base_size)

        f4, a4, _ = self.attention(f2, x_t, a2, coarse_uncertainty)
        out4 = self.res(a4, base_size)

        out = out1 + out2 + out3 + out4

        refinement_feat = torch.cat([f3, f4], dim=1)

        band_pred = self.band_head(refinement_feat)
        band_pred_up = self.res(band_pred, base_size)

        background_pred = self.background_head(refinement_feat)
        background_pred_up = self.res(background_pred, base_size)

        uncertainty_full = self.res(coarse_uncertainty, base_size)
        uncertainty_full = self.normalize_uncertainty(uncertainty_full)

        band_target = self.make_band_target(y, kernel_size=7)

        pixel_weight = 1.0 + 1.25 * uncertainty_full + 0.5 * band_target

        loss1 = self.seg_loss(out1, y)
        loss2 = self.seg_loss(out2, y)
        loss3 = self.seg_loss(out3, y)
        loss4 = self.seg_loss(out4, y)

        loss_final = self.seg_loss(out, y)

        loss_precision_tversky = self.precision_tversky_loss(out, y)

        loss_consistency = self.selective_agreement_loss(
            out1,
            out2,
            uncertainty_full
        )

        loss_band = self.weighted_seg_loss(
            band_pred_up,
            band_target,
            pixel_weight=pixel_weight
        )

        loss_conservative_ugel = self.conservative_uncertainty_gated_edge_loss(
            logits=out,
            mask=y,
            sample=sample
        )

        loss_false_positive = self.false_positive_suppression_loss(
            logits=out,
            mask=y,
            sample=sample
        )

        loss_background = self.background_supervision_loss(
            background_logits=background_pred_up,
            mask=y,
            sample=sample
        )

        loss_area = self.area_regularization_loss(
            logits=out,
            mask=y
        )

        loss = (
            self.loss_w1 * loss1
            + self.loss_w2 * loss2
            + self.loss_w3 * loss3
            + self.loss_w4 * loss4
            + self.final_loss_weight * loss_final
            + self.precision_tversky_weight * loss_precision_tversky
            + self.consistency_weight * loss_consistency
            + self.band_loss_weight * loss_band
            + self.conservative_ugel_weight * loss_conservative_ugel
            + self.false_positive_weight * loss_false_positive
            + self.background_loss_weight * loss_background
            + self.area_loss_weight * loss_area
        )

        zero_loss = loss_final.detach() * 0.0

        return {
            "prediction": out,
            "loss": loss,

            "loss_final": loss_final.detach(),
            "loss_focal_tversky": loss_precision_tversky.detach(),
            "loss_band": loss_band.detach(),
            "loss_soft_ugel": loss_conservative_ugel.detach(),

            # Kept for compatibility with your existing train scripts.
            "loss_freq_boundary": zero_loss,

            "loss_consistency": loss_consistency.detach(),

            # Extra debug values. Existing train scripts can ignore them.
            "loss_false_positive": loss_false_positive.detach(),
            "loss_background": loss_background.detach(),
            "loss_area": loss_area.detach()
        }

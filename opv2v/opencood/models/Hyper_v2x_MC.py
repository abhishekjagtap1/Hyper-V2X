"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import math

class ProbHyperNet(nn.Module):
    """Hypernetwork generating mean and logvar for conv params."""
    def __init__(self, cond_dim, output_param_count, hidden_sizes=(256,256)):
        super().__init__()
        layers = []
        prev = cond_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_param_count*2))
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        out = self.net(cond)
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar


class HyperSegHead(nn.Module):
    """
    Hypernetwork-powered segmentation head.
    Samples conv3x3 weights conditioned on fused features.
    """
    def __init__(self, in_channels, num_classes, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        # shapes for conv3x3
        self.weight_shape = (num_classes, in_channels, kernel_size, kernel_size)
        self.bias_shape = (num_classes,)
        w_count = int(torch.tensor(self.weight_shape).prod())
        b_count = int(torch.tensor(self.bias_shape).prod())
        self.total_params = w_count + b_count

        self.hyper = ProbHyperNet(cond_dim=in_channels,
                                  output_param_count=self.total_params)

        self.register_buffer("prior_var", torch.tensor(1.0))

    def _unflatten(self, vec):
        B = vec.shape[0]
        idx = 0
        w_num = int(torch.tensor(self.weight_shape).prod())
        w = vec[:, idx:idx+w_num]
        idx += w_num
        b_num = int(torch.tensor(self.bias_shape).prod())
        b = vec[:, idx:idx+b_num]
        w = w.view(B, *self.weight_shape)
        b = b.view(B, *self.bias_shape)
        return {"w": w, "b": b}

    def sample_params(self, cond, K=4):
        mean, logvar = self.hyper(cond)
        var = torch.exp(logvar)

        prior_var = float(self.prior_var.item())
        kl = 0.5 * ((mean**2 + var)/prior_var - 1 - torch.log(var+1e-9) + math.log(prior_var))
        kl = kl.sum(dim=-1).mean()

        eps = torch.randn(K, mean.shape[0], self.total_params, device=mean.device)
        std = torch.sqrt(var+1e-9)
        samples = mean.unsqueeze(0) + eps*std.unsqueeze(0)

        param_samples = []
        for k in range(K):
            param_samples.append(self._unflatten(samples[k]))

        stacked = {k: torch.stack([p[k] for p in param_samples], dim=0)
                   for k in param_samples[0].keys()}
        return stacked, kl

    def forward_with_params(self, feat, params):
        K, B = params["w"].shape[:2]
        C, H, W = feat.shape[1:]
        out_ch = self.num_classes

        w = params["w"].permute(1,0,2,3,4,5).contiguous()
        w = w.view(B*(K*out_ch), C, self.kernel_size, self.kernel_size)
        x = feat.reshape(1, B*C, H, W)
        out = F.conv2d(x, w, bias=None, stride=1, padding=self.kernel_size//2, groups=B)
        out = out.view(B, K, out_ch, H, W)
        out = out.permute(1,0,2,3,4).contiguous()
        out = out + params["b"].unsqueeze(-1).unsqueeze(-1)
        return out  # [K,B,C,H,W]

    def forward(self, feat, K=4):
        cond = feat.mean(dim=[2,3])  # [B,C]
        params, kl = self.sample_params(cond, K)
        mc_out = self.forward_with_params(feat, params)  # [K,B,C,H,W]
        mean = mc_out.mean(0)
        var = mc_out.var(0, unbiased=False)
        return mean, var, kl

class HyperBevSegHead(nn.Module):
    """
    Hypernetwork-based BEV segmentation head that matches the vanilla interface.
    Supports 'dynamic', 'static', or both targets.
    """
    def __init__(self, target, in_channels, num_classes):
        super().__init__()
        self.target = target

        if self.target == "dynamic":
            self.dynamic_head = HyperSegHead(in_channels, num_classes)
            self.static_head = None
        elif self.target == "static":
            self.static_head = HyperSegHead(in_channels, num_classes)
            self.dynamic_head = None
        else:
            self.dynamic_head = HyperSegHead(in_channels, num_classes)
            self.static_head = HyperSegHead(in_channels, num_classes)

    def forward(self, x, b, l, K=4):
        out = {}

        # Dynamic-only
        if self.target == "dynamic":
            dyn_mean, dyn_var, dyn_kl = self.dynamic_head(x, K)
            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = torch.zeros_like(dyn_mean)
            out["dynamic_var"] = dyn_var
            out["kl"] = dyn_kl

        # Static-only
        elif self.target == "static":
            stat_mean, stat_var, stat_kl = self.static_head(x, K)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["static_seg"] = stat_mean
            out["dynamic_seg"] = torch.zeros_like(stat_mean)
            out["static_var"] = stat_var
            out["kl"] = stat_kl

        # Both dynamic and static
        else:
            dyn_mean, dyn_var, dyn_kl = self.dynamic_head(x, K)
            stat_mean, stat_var, stat_kl = self.static_head(x, K)

            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = stat_mean
            out["dynamic_var"] = dyn_var
            out["static_var"] = stat_var
            out["kl"] = 0.5 * (dyn_kl + stat_kl)

        return out



class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = HyperBevSegHead(self.target,
                                        config['seg_head_dim'],
                                        config['output_class'])

        #self.seg_head = BevSegHead(self.target,
         #                          config['seg_head_dim'],
          #                         config['output_class'])

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.fax(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        # compressor
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w') # 1, 32, 256, 256
        b = x.shape[0]

        output_dict = self.seg_head(x, b, 1) # 1, 1, 2, 256, 256 vanilla head

        return output_dict
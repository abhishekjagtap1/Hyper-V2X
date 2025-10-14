"""
Implementation of GyperNetwork inspired HyperDM: Estimating Both uncertainity in single model
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
from einops import rearrange
import math


"""
This model is used to train Hyper_V2X_IV folder results in logs
1) Added Probabilities for epistemic var since logits was not accurate 
2) Added Alleotary uncertainity support 
3) Total Uncertainity
4) The model outputs 5 items: Pred_mean_seg, Epistemic, alleotary, Total_uncertainity and KL
5) Note this is used in conjucture with loss function stored at Hyper_V2X_loss
6) Trained using K = 4, now little flexible to use K


Inference:
1) Support for visulization of all uncertainity required adopt the code in shata_inference.py to do it
2) Lets see if I can make it this time

"""

# -------------------------
# Bayesian Hypernetwork
# -------------------------
class VariationalMultiHyperNet(nn.Module):
    """
    Outputs posterior mean and logvar for decoder parameters.
    For stability we predict logvar and clamp it.
    """
    def __init__(self, cond_dim, output_param_count, hidden_sizes=(256,256)):
        super().__init__()
        layers = []
        prev = cond_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # we output mean and logvar concatenated: 2*P
        layers.append(nn.Linear(prev, output_param_count * 2))
        self.net = nn.Sequential(*layers)
        self._logvar_min = -20.0
        self._logvar_max = 5.0

    def forward(self, cond):
        """
        cond: [B, cond_dim]
        returns: mu [B, P], logvar [B, P]
        """
        out = self.net(cond)  # [B, 2*P]
        B = cond.shape[0]
        P2 = out.shape[-1]
        P = P2 // 2
        mu = out[:, :P]
        logvar = out[:, P:]
        # numerical stability clamp
        logvar = torch.clamp(logvar, self._logvar_min, self._logvar_max)
        return mu, logvar



# -------------------------
# Segmentation Head
# -------------------------
class HyperSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, K, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.weight_shape = (num_classes, in_channels, kernel_size, kernel_size)
        self.bias_shape = (num_classes,)
        w_count = int(torch.tensor(self.weight_shape).prod())
        b_count = int(torch.tensor(self.bias_shape).prod())
        self.total_params = w_count + b_count
        self.K = K


        # use variational hypernet
        self.hyper = VariationalMultiHyperNet(cond_dim=in_channels,
                                              output_param_count=self.total_params)

    def _unflatten(self, vec):
        # vec: [B, P] or [B, K, P] depending on input
        if vec.dim() == 2:
            B, P = vec.shape
            vec = vec.unsqueeze(1)  # -> [B,1,P]
        B, K, P = vec.shape
        idx = 0
        w_num = int(torch.tensor(self.weight_shape).prod())
        w = vec[:, :, idx:idx+w_num]; idx += w_num
        b_num = int(torch.tensor(self.bias_shape).prod())
        b = vec[:, :, idx:idx+b_num]
        w = w.view(B, K, *self.weight_shape)   # [B,K,C_out,C_in,H,W]
        b = b.view(B, K, *self.bias_shape)     # [B,K,C_out]
        return {"w": w, "b": b}

    def _sample_params(self, mu, logvar, K):
        """
        mu, logvar: [B, P]
        returns sampled vec: [B, K, P] and KL per sample (or per-batch KL)
        """
        B, P = mu.shape
        std = torch.exp(0.5 * logvar)
        # sample eps: [B, K, P]
        eps = torch.randn(B, K, P, device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)  # [B, K, P]

        # KL between diagonal Gaussian and standard normal prior (per batch)
        # KL per element: 0.5*(mu^2 + sigma^2 - logvar - 1)
        kl_per_dim = 0.5 * (mu**2 + std**2 - logvar - 1.0)  # [B, P]
        # sum dims -> KL per sample (same for each K since q params shared across K)
        kl = kl_per_dim.sum(dim=1)  # [B]
        # return samples and mean KL per batch element
        return samples, kl  # samples: [B,K,P], kl: [B]

    def forward_with_params(self, feat, params):
        # identical to your previous code, expects params["w"]: [B,K,C_out,C_in,H,W]
        K, B = params["w"].shape[1], params["w"].shape[0]
        C, H, W = feat.shape[1:]
        out_ch = self.num_classes

        # rearrange weights to grouped conv
        w = params["w"].reshape(B*(K*out_ch), C, self.kernel_size, self.kernel_size)
        x = feat.reshape(1, B*C, H, W)
        out = F.conv2d(x, w, bias=None, stride=1, padding=self.kernel_size//2, groups=B)
        out = out.view(B, K, out_ch, H, W)
        out = out.permute(1,0,2,3,4).contiguous()  # [K,B,C,H,W]
        out = out + params["b"].permute(1,0,2).unsqueeze(-1).unsqueeze(-1)  # add bias
        return out

    def forward(self, feat, K, beta=1e-3):
        """
        feat: [B, C, H, W]
        returns: predictive mean [B,C,H,W], predictive var [B,C,H,W], kl [B]
        """
        cond = feat.mean(dim=[2,3])  # [B,C]
        B, C, H, W = feat.shape
        #cond = torch.randn(B, self.in_channels, device=feat.device, dtype=feat.dtype)
        mu, logvar = self.hyper(cond)  # [B,P], [B,P]
        samples, kl_per_batch = self._sample_params(mu, logvar, K)  # [B,K,P], [B]
        params = self._unflatten(samples)  # {"w": [B,K,C_out,C_in,H,W], "b": [B,K,C_out]}

        outs = self.forward_with_params(feat, params)  # [K,B,C,H,W]
        # compute predictive mean and variance over samples dimension
        pred_mean = outs.mean(0)   # [B,C,H,W]


        # Both Epistemic and alleotary uncertainity
        prob_outs =  torch.softmax(outs, dim=2)
        
        epistemic_unc = prob_outs.var(0, unbiased=False)  # [B,C,H,W]
        aleatoric_unc = -(prob_outs.mean(0) * torch.log(prob_outs.mean(0)  + 1e-8)).sum(dim=1)  # [B,H,W]
        aleatoric_unc = aleatoric_unc.unsqueeze(1)  # [B,1,H,W]

        # average kl over batch to return scalar per batch or keep per-sample
        kl = kl_per_batch.mean()
        total_unc = epistemic_unc + aleatoric_unc  # scalar; you may prefer sum or mean depending on loss scaling

        return pred_mean, epistemic_unc, aleatoric_unc, total_unc, kl



# -------------------------
# BEV Segmentation Head
# -------------------------
class HyperBevSegHead(nn.Module):
    """
    Hypernetwork-based BEV segmentation head that matches the vanilla interface.
    Supports 'dynamic', 'static', or both targets.
    """
    def __init__(self, target, in_channels, num_classes, K):
        super().__init__()
        self.target = target
        self.K = K

        if self.target == "dynamic":
            self.dynamic_head = HyperSegHead(in_channels, num_classes, K)
            self.static_head = None
        elif self.target == "static":
            self.static_head = HyperSegHead(in_channels, num_classes, K)
            self.dynamic_head = None
        else:
            self.dynamic_head = HyperSegHead(in_channels, num_classes, K)
            self.static_head = HyperSegHead(in_channels, num_classes, K)

    def forward(self, x, b, l, K=None):
        out = {}

        # Dynamic-only
        if self.target == "dynamic":
            dyn_mean, dyn_var, dyn_aleo, total_unc, dyn_kl = self.dynamic_head(x, K)
            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_aleo = rearrange(dyn_aleo, "(b l) c h w -> b l c h w", b=b, l=l)
            total_unc = rearrange(total_unc, "(b l) c h w -> b l c h w", b=b, l=l)

            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = torch.zeros_like(dyn_mean)
            out["dynamic_var"] = dyn_var
            out["dynamic_aleo"] = dyn_aleo
            out["total_unc"] = total_unc
            out["kl"] = dyn_kl

        # Static-only
        elif self.target == "static":
            stat_mean, stat_var, stat_aleo, stat_kl = self.static_head(x, K)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_aleo = rearrange(stat_aleo, "(b l) c h w -> b l c h w", b=b, l=l)

            out["static_seg"] = stat_mean
            out["dynamic_seg"] = torch.zeros_like(stat_mean)
            out["static_var"] = stat_var
            out["static_aleo"] = stat_aleo
            out["kl"] = stat_kl

        # Both dynamic and static
        else:
            dyn_mean, dyn_var, dyn_aleo, dyn_kl = self.dynamic_head(x, K)
            stat_mean, stat_var, stat_aleo, stat_kl = self.static_head(x, K)

            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_aleo = rearrange(dyn_aleo, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_aleo = rearrange(stat_aleo, "(b l) c h w -> b l c h w", b=b, l=l)


            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = stat_mean
            out["dynamic_var"] = dyn_var
            out["static_var"] = stat_var
            out["static_aleo"] = stat_aleo
            out["dynamic_aleo"] = dyn_aleo
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
        ##################################################################################################
        self.K = 4
        ###############################################################################################
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
                                        config['output_class'],
                                        self.K)

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

        output_dict = self.seg_head(x, b, 1, self.K) # 1, 1, 2, 256, 256 vanilla head

        return output_dict
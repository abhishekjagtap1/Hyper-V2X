
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch

class MultiHyperNet(nn.Module):
    """Hypernetwork generating K independent sets of conv params."""
    def __init__(self, cond_dim, output_param_count, K=4, hidden_sizes=(256,256)):
        super().__init__()
        self.K = K
        layers = []
        prev = cond_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_param_count * K))
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        out = self.net(cond)  # [B, K*P]
        B = cond.shape[0]
        out = out.view(B, self.K, -1)  # [B,K,P]
        return out  # deterministic K parameter vectors


class HyperDecoder(nn.Module):
    def __init__(self, params, cond_dim, K=4):
        super().__init__()
        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']
        self.K = K

        # Collect all conv param counts
        self.param_shapes = []
        for i in range(self.num_layer-1, -1, -1):
            num_ch_in = self.input_dim if i == self.num_layer-1 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            # upconv_0
            w0_shape = (num_ch_out, num_ch_in, 3, 3)
            b0_shape = (num_ch_out,)
            self.param_shapes.append((w0_shape, b0_shape))

            # upconv_1
            w1_shape = (num_ch_out, num_ch_out, 3, 3)
            b1_shape = (num_ch_out,)
            self.param_shapes.append((w1_shape, b1_shape))

        # Flatten count
        self.total_params = sum(
            torch.tensor(w).prod().item() + torch.tensor(b).prod().item()
            for w, b in self.param_shapes
        )

        # One hypernet outputs everything
        self.hyper = MultiHyperNet(cond_dim, self.total_params, K=K)

        # BN + ReLU stay as modules
        self.norms = nn.ModuleList([nn.BatchNorm2d(s[0][0]) for s in self.param_shapes])
        self.relus = nn.ModuleList([nn.ReLU(True) for _ in self.param_shapes])

    def _unflatten(self, vec):
        # vec: [B,K,P]
        B, K, _ = vec.shape
        params = []
        idx = 0
        for w_shape, b_shape in self.param_shapes:
            w_num = torch.tensor(w_shape).prod().item()
            b_num = torch.tensor(b_shape).prod().item()

            w = vec[:, :, idx:idx+w_num].view(B, K, *w_shape)
            idx += w_num
            b = vec[:, :, idx:idx+b_num].view(B, K, *b_shape)
            idx += b_num
            params.append((w, b))
        return params  # list of (w, b) for each conv

    def forward_with_params(self, feat, params):
        B, C, H, W = feat.shape
        x = feat
        layer_idx = 0

        for i in range(self.num_layer-1, -1, -1):
            # conv0
            w, b = params[layer_idx]
            layer_idx += 1
            x = self._apply_conv(x, w, b)
            x = self.norms[layer_idx-1](x)
            x = self.relus[layer_idx-1](x)

            x = F.interpolate(x, scale_factor=2, mode="nearest")

            # conv1
            w, b = params[layer_idx]
            layer_idx += 1
            x = self._apply_conv(x, w, b)
            x = self.norms[layer_idx-1](x)
            x = self.relus[layer_idx-1](x)

        return x

    def _apply_conv(self, x, w, b):
        """
        x: [B,C,H,W], w: [B,K,C_out,C_in,k,k], b: [B,K,C_out]
        We can either pick 1 K or loop over K.
        """
        # For now: just pick mean over K
        w = w.mean(1)  # [B,C_out,C_in,k,k]
        b = b.mean(1)  # [B,C_out]
        out = F.conv2d(x, w, b, stride=1, padding=1)
        return out

    def forward(self, feat):
        cond = feat.mean(dim=[2,3])  # [B,C]
        vecs = self.hyper(cond)      # [B,K,P]
        params = self._unflatten(vecs)
        out = self.forward_with_params(feat, params)
        return out

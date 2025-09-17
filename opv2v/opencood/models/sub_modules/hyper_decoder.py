# hyper_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHyperNet(nn.Module):
    """Deterministic hypernet that outputs K slots in one forward pass.
       Input: cond tensor shape [N, cond_dim] where N = B*L (flattened samples).
       Output: [N, K, P] where P = total param count to predict.
    """
    def __init__(self, cond_dim, output_param_count, K=4, hidden_sizes=(256,256)):
        super().__init__()
        self.K = K
        self.cond_dim = cond_dim
        layers = []
        prev = cond_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_param_count * K))
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        # cond: [N, cond_dim]
        out = self.net(cond)       # [N, K*P]
        N = cond.shape[0]
        out = out.view(N, self.K, -1)  # [N, K, P]
        return out


class HyperDecoder(nn.Module):
    """
    HyperDecoder: replaces NaiveDecoder and predicts conv weights for all decoder convs.
    Kept BatchNorm/ReLU as normal modules (not predicted).
    forward(x) expects x: [B, L, C, H, W] and returns [B, L, C_out, H_out, W_out].
    Robust to cond_dim mismatches: if cond_dim passed to constructor != actual input C,
    a small linear projection (cond_proj) maps actual cond -> hypernet_cond_dim.
    """
    def __init__(self, params, cond_dim=None, K=4, hidden_sizes=(256,256)):
        super().__init__()
        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']  # expected channel size of input features
        self.K = K

        # Build list of conv param shapes in the same order as NaiveDecoder:
        self.param_shapes = []
        for i in range(self.num_layer - 1, -1, -1):
            num_ch_in = self.input_dim if i == self.num_layer - 1 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # upconv_0
            self.param_shapes.append(((num_ch_out, num_ch_in, 3, 3), (num_ch_out,)))
            # upconv_1
            self.param_shapes.append(((num_ch_out, num_ch_out, 3, 3), (num_ch_out,)))

        # total number of scalar params (weights + biases)
        total = 0
        for w_shape, b_shape in self.param_shapes:
            w_count = 1
            for d in w_shape:
                w_count *= d
            b_count = 1
            for d in b_shape:
                b_count *= d
            total += w_count + b_count
        self.total_params = total

        # Decide hypernet cond dim:
        # - if cond_dim is None => use decoder input_dim (no projection needed)
        # - if cond_dim provided and differs from input_dim => create cond_proj
        if cond_dim is None:
            hyper_cond_dim = self.input_dim
            self.cond_proj = None
        else:
            hyper_cond_dim = cond_dim
            if cond_dim != self.input_dim:
                # create a small projection to map actual cond (input_dim) -> hypernet_cond_dim
                self.cond_proj = nn.Linear(self.input_dim, hyper_cond_dim)
            else:
                self.cond_proj = None

        # create hypernet
        self.hyper = MultiHyperNet(hyper_cond_dim, self.total_params, K=self.K, hidden_sizes=hidden_sizes)

        # keep BN/ReLU modules (not predicted)
        self.norms = nn.ModuleList([nn.BatchNorm2d(w_shape[0]) for (w_shape, _) in self.param_shapes])
        self.relus = nn.ModuleList([nn.ReLU(True) for _ in self.param_shapes])

    def _unflatten(self, vec):
        """
        vec: [N, K, P] -> returns list of (w, b) where:
            w: [N, K, out_ch, in_ch, kh, kw]
            b: [N, K, out_ch]
        """
        N, K, P = vec.shape
        params = []
        idx = 0
        for w_shape, b_shape in self.param_shapes:
            # counts
            w_count = 1
            for d in w_shape:
                w_count *= d
            b_count = 1
            for d in b_shape:
                b_count *= d

            w_flat = vec[:, :, idx: idx + w_count]
            idx += w_count
            b_flat = vec[:, :, idx: idx + b_count]
            idx += b_count

            w = w_flat.view(N, K, *w_shape)  # [N, K, out_ch, in_ch, kh, kw]
            b = b_flat.view(N, K, *b_shape)   # [N, K, out_ch]
            params.append((w, b))
        return params

    def forward(self, x):
        """
        x: [B, L, C, H, W]
        returns: [B, L, C_out, H_out, W_out]
        """
        b, l, c, h, w = x.shape
        N = b * l
        x_flat = x.reshape(N, c, h, w)          # [N, C, H, W]

        # conditioning vector (mean pooling spatially)
        cond = x_flat.mean(dim=[2, 3])          # [N, C]

        # optionally project cond to hypernet cond_dim if needed
        if self.cond_proj is not None:
            cond = self.cond_proj(cond)         # [N, hyper_cond_dim]

        # sanity check
        if cond.shape[1] != self.hyper.cond_dim:
            raise RuntimeError(
                f"HyperDecoder: conditioning dim mismatch after projection "
                f"(got {cond.shape[1]}, hyper expects {self.hyper.cond_dim})"
            )

        # hypernet predicts [N, K, P]
        vecs = self.hyper(cond)                 # [N, K, P]
        params = self._unflatten(vecs)          # list of (w,b) per conv in order

        # compute K decoded outputs (one per slot) and average
        outputs_per_k = []
        for k in range(self.K):
            xk = x_flat   # start with [N, C, H, W]
            layer_idx = 0
            Hcur, Wcur = h, w

            for (w_shape, b_shape) in self.param_shapes:
                w_all = params[layer_idx][0][:, k]   # [N, out_ch, in_ch, kh, kw]
                b_all = params[layer_idx][1][:, k]   # [N, out_ch]

                N_local = xk.shape[0]
                in_ch = xk.shape[1]
                kh = w_all.shape[-2]
                kw = w_all.shape[-1]

                # reshape weights -> (N*out_ch, in_ch, kh, kw)
                weight_view = w_all.reshape(N_local * w_all.shape[1], w_all.shape[2], kh, kw).contiguous()

                # reshape input -> (1, N*in_ch, H, W)
                x_in = xk.reshape(1, N_local * in_ch, Hcur, Wcur)

                out = F.conv2d(x_in,
                               weight_view,
                               bias=None,
                               stride=1,
                               padding=kh // 2,
                               groups=N_local)

                # -> (1, N*out_ch, H, W) -> (N, out_ch, H, W)
                out = out.view(N_local, w_all.shape[1], Hcur, Wcur)
                out = out + b_all.unsqueeze(-1).unsqueeze(-1)    # add per-sample bias

                # BN + ReLU
                out = self.norms[layer_idx](out)
                out = self.relus[layer_idx](out)

                # upsample after conv0 of pair (layer_idx even)
                if (layer_idx % 2) == 0:
                    out = F.interpolate(out, scale_factor=2, mode='nearest')
                    Hcur = out.shape[2]
                    Wcur = out.shape[3]

                xk = out
                layer_idx += 1

            outputs_per_k.append(xk)

        stack = torch.stack(outputs_per_k, dim=0)    # [K, N, C_out, H_out, W_out]
        meanN = stack.mean(dim=0)                    # [N, C_out, H_out, W_out]

        out_final = meanN.view(b, l, meanN.shape[1], meanN.shape[2], meanN.shape[3])
        return out_final

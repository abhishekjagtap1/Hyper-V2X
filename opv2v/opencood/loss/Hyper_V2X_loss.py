import torch
import torch.nn as nn

from einops import rearrange


class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.target = args['target']
        self.beta = 1e-3

        self.loss_func_static = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

        self.loss_dict = {}

    def forward(self, output_dict, gt_dict):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """

        # Outputs from Bayesian hypernet forward
        static_pred = output_dict['static_seg']  # [B, L, C, H, W]
        dynamic_pred = output_dict['dynamic_seg']  # [B, L, C, H, W]
        kl = output_dict['kl']  # scalar or [B]

        static_loss = torch.tensor(0., device=static_pred.device)
        dynamic_loss = torch.tensor(0., device=dynamic_pred.device)

        # Ground truth
        static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

        # ---- Negative log likelihood (CrossEntropy) ----
        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            per_pixel_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            dynamic_loss = per_pixel_loss.mean()

        elif self.target == 'static':
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            per_pixel_loss = self.loss_func_static(static_pred, static_gt)
            static_loss = per_pixel_loss.mean()

        else:  # both
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')

            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt).mean()
            static_loss = self.loss_func_static(static_pred, static_gt).mean()

        # ---- ELBO objective ----
        total_nll = self.s_coe * static_loss + self.d_coe * dynamic_loss
        total_loss = total_nll + self.beta * kl

        # Save for logging
        self.loss_dict.update({
            'total_loss': total_loss,
            'static_loss': static_loss,
            'dynamic_loss': dynamic_loss,
            'KL_loss': kl,
            'NLL_loss': total_nll
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']
        kl_loss = self.loss_dict['KL_loss']
        nll_loss = self.loss_dict['NLL_loss']

        if pbar is None:
            print(f"[epoch {epoch}][{batch_id + 1}/{batch_len}] || "
                  f"Loss: {total_loss:.4f} || NLL: {nll_loss:.4f} || KL: {kl_loss:.4f} || "
                  f"Static: {static_loss:.4f} || Dynamic: {dynamic_loss:.4f}")
        else:
            pbar.set_description(f"[epoch {epoch}][{batch_id + 1}/{batch_len}] || "
                                 f"Loss: {total_loss:.4f} || NLL: {nll_loss:.4f} || KL: {kl_loss:.4f} || "
                                 f"Static: {static_loss:.4f} || Dynamic: {dynamic_loss:.4f}")

        writer.add_scalar('Total_loss', total_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('NLL_loss', nll_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('KL_loss', kl_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('Static_loss', static_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                          epoch * batch_len + batch_id)



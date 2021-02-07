import torch
from ckconv.nn import CKConv


class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.

        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super(LnLoss, self).__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: CKConv,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs and gather the sampled filters
        for m in model.modules():
            if not isinstance(m, CKConv):
                continue
            loss = loss + m.conv_kernel.norm(self.norm_type)
            loss = loss + m.bias.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss

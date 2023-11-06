import torch
import torch.nn as nn
import torch.nn.functional as F


def get_negative_mask(batch_size):
    mask1 = torch.ones((batch_size, 2 * batch_size - 1), dtype=bool)
    for i in range(batch_size):
        mask1[i, i + batch_size - 1] = 0
    mask2 = torch.ones((batch_size, 2 * batch_size - 1), dtype=bool)
    for i in range(batch_size):
        mask2[i, i] = 0
    mask = torch.cat((mask1, mask2), 0)
    return mask


def get_label(batch_size):
    mask = torch.zeros((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        mask[i, i] = 1
        mask[i, i + batch_size] = 1
    positive_mask = torch.cat((mask, mask), 0)
    positive_mask[:batch_size, :batch_size] = 0
    positive_mask[batch_size:, batch_size:] = 0
    label = torch.nonzero(positive_mask)
    for i in range(batch_size):
        label[i] = label[i] - 1
    return label[:, 1]


def get_posotive_mask(batch_size):
    positive_mask = torch.zeros((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        positive_mask[i, i] = 1
        positive_mask[i, i + batch_size] = 1

    positive_mask = torch.cat((positive_mask, positive_mask), 0)
    positive_mask[:batch_size, :batch_size] = 0
    positive_mask[batch_size:, batch_size:] = 0
    return positive_mask


class CurricularLoss(nn.Module):
    def __init__(self, device, momentum, regular, if_cuda=False):
        super(CurricularLoss, self).__init__()
        self.register_buffer('t', torch.zeros(1))
        self.if_cuda = if_cuda
        self.device = device
        self.momentum = momentum
        self.regular = regular

    def forward(self, out_1, out_2, batch_size, temperature):
        out = torch.cat([out_1, out_2], dim=0)
        cos_theta = torch.mm(out, out.t())
        cos_theta = cos_theta.clamp(-1, 1)

        pos_mask = get_posotive_mask(batch_size)
        if self.if_cuda:
            pos_mask = pos_mask.to(self.device)
        target_logit = cos_theta[pos_mask].view(-1, 1)

        mask = (torch.ones_like(cos_theta) - torch.eye(2 * batch_size, device=cos_theta.device)).bool()
        cos_theta = cos_theta.masked_select(mask).view(2 * batch_size, -1)

        mask_neg = get_negative_mask(batch_size)
        if self.if_cuda:
            mask_neg = mask_neg.to(self.device)
        cos_theta_neg = cos_theta.masked_select(mask_neg).view(2 * batch_size, -1)

        _, index = torch.sort(cos_theta_neg, 1, True)

        label = get_label(batch_size)
        if self.if_cuda:
            label = label.to(self.device)

        # update t
        with torch.no_grad():
            self.t = self.momentum * self.t + target_logit.mean() * (1 - self.momentum)

        mask = cos_theta > target_logit
        hard = cos_theta.masked_fill(~mask, -self.t[0])
        weight = hard + self.t
        mask_weight = weight > 1
        # regularization
        regular = weight.masked_fill(~mask_weight, 0)
        re = 0.5 * self.regular * torch.sum(torch.pow(regular, 2), dim=1)
        # negative similarity
        hard_example = cos_theta[mask]
        cos_theta[mask] = hard_example * (self.t + hard_example)
        # calculate loss
        output = cos_theta / temperature
        loss = F.cross_entropy(output, label) + re.sum()

        return loss

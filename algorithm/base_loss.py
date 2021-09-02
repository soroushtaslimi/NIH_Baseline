import torch
import torch.nn.functional as F
import numpy as np


def loss_base(logits, labels, forget_rate=None, class_weights=None):
    labels = labels.type_as(logits)  # for bceloss type of t should be float
    loss_pick_1 = F.binary_cross_entropy_with_logits(logits, labels, reduction='none', weight=class_weights)

    if forget_rate is None:
        loss = torch.mean(loss_pick_1)
    else:
        loss_pick_1 = torch.mean(loss_pick_1, dim=1)

        # loss_pick = loss_pick_1.cpu()

        ind_sorted = np.argsort(loss_pick_1.data)
        loss_sorted = loss_pick_1[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]

        loss = torch.mean(loss_pick_1[ind_update])

    return loss

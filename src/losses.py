import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        ce = F.nll_loss(logp, target, reduction='none')
        modulating = (1 - p.gather(1, target.unsqueeze(1)).squeeze(1)) ** self.gamma
        loss = modulating * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class SuperLoss(nn.Module):
    def __init__(self, num_classes, momentum=0.95, base_temp=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.base_temp = base_temp
        self.register_buffer('class_centers', torch.zeros(num_classes))
        self.register_buffer('class_counts', torch.ones(num_classes))
        self.temperature = nn.Parameter(torch.ones(num_classes) * base_temp)

    def forward(self, logits, targets):
        device = logits.device
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            one_hot = F.one_hot(targets, self.num_classes).float().to(device)
            counts = one_hot.sum(0)
            self.class_counts = self.momentum * self.class_counts + (1 - self.momentum) * counts
            self.class_centers = self.momentum * self.class_centers + (1 - self.momentum) * probs.mean(0)
        temp = self.base_temp * (self.class_counts / self.class_counts.mean()).detach()
        adjusted = logits / self.temperature.unsqueeze(0).to(device)
        loss = F.cross_entropy(adjusted, targets, reduction='none')
        return loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=None, reduction='none')
        self.smoothing = smoothing
        self.ce_smooth = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, logits, target):
        fl = self.focal(logits, target)
        ls = self.ce_smooth(logits, target)
        return 0.5 * fl + 0.5 * ls

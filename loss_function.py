import torch
from torch import nn
from torch.nn import functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, mel_lens, dur=None):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        _, mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        loss = mel_loss + gate_loss
        return loss, (loss.data.cpu().numpy())


class ForwardTacotronLoss(torch.nn.Module):
    def forward_one(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(1)
        mask = pad_mask(lens, max_len)
        mask = mask.unsqueeze(2).expand_as(x)
        loss = F.mse_loss(
            x * mask, target * mask, reduction='sum')
        return loss / mask.sum()

    def forward(self, model_outputs, targets, mel_lens, dur):
        m1, m2, dur_hat = model_outputs
        target, *_ = targets

        m1_loss = self.forward_one(m1, target, mel_lens)
        m2_loss = self.forward_one(m2, target, mel_lens)
        dur_loss = F.l1_loss(dur_hat, dur)
        loss = m1_loss + m2_loss + dur_loss
        return loss, (m1_loss.data.cpu().numpy(), m2_loss.data.cpu().numpy(), dur_loss.data.cpu().numpy())


# Adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def pad_mask(lens, max_len):
    batch_size = lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range = seq_range.unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    if lens.is_cuda:
        seq_range = seq_range.cuda()
    lens = lens.unsqueeze(1)
    lens = lens.expand_as(seq_range)
    mask = seq_range < lens
    return mask.float()

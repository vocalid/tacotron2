import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class ForwardTacotronLogger(SummaryWriter):
    def __init__(self, logdir):
        super(ForwardTacotronLogger, self).__init__(logdir)

    def log_training(self, total_loss, taco_loss, mi_loss, grad_norm,
                     gaf, learning_rate, duration, iteration):
            self.add_scalar("fwdtaco/training.loss", total_loss, iteration)
            self.add_scalar("fwdtaco/training.taco_loss", taco_loss, iteration)
            #self.add_scalar("fwdtaco/training.mi_loss", mi_loss, iteration)
            self.add_scalar("fwdtaco/grad.norm", grad_norm, iteration)
            #self.add_scalar("fwdtaco/grad.gaf", gaf, iteration)
            self.add_scalar("fwdtaco/learning.rate", learning_rate, iteration)
            self.add_scalar("fwdtaco/duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        #_, _, mel_outputs, gate_outputs, alignments = y_pred
        m1, m2, dur_hat = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        #idx = random.randint(0, alignments.size(0) - 1)
        idx = random.randint(0, mel_targets.size(0) - 1)
        #self.add_image(
        #    "alignment",
        #    plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        #    iteration, dataformats='HWC')
        self.add_image(
            "forwardtaco/mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "forwardtaco/mel_predicted_pre",
            plot_spectrogram_to_numpy(m1[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "forwardtaco/mel_predicted_post",
            plot_spectrogram_to_numpy(m2[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        #self.add_image(
        #    "gate",
        #    plot_gate_outputs_to_numpy(
        #        gate_targets[idx].data.cpu().numpy(),
        #        torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        #    iteration, dataformats='HWC')


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, total_loss, taco_loss, mi_loss, grad_norm,
                     gaf, learning_rate, duration, iteration):
            self.add_scalar("training.loss", total_loss, iteration)
            self.add_scalar("training.taco_loss", taco_loss, iteration)
            self.add_scalar("training.mi_loss", mi_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("grad.gaf", gaf, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, _, mel_outputs, gate_outputs, alignments = y_pred
        m1, m2, dur_hat = model_outputs
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

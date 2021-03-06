import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

import tacorn.log

slack_logger = tacorn.log.get_slack_log()
slack_log_interval = 1000

class ForwardTacotronLogger(SummaryWriter):
    def __init__(self, logdir):
        super(ForwardTacotronLogger, self).__init__(logdir)
        self.last_slack_log = 0

    def log_training(self, loginfo, total_loss, taco_loss, mi_loss, grad_norm,
                     gaf, learning_rate, duration, iteration):
            m1_loss, m2_loss, dur_loss = loginfo
            self.add_scalar("forwardtaco/training.loss", total_loss, iteration)
            self.add_scalar("forwardtaco/training.taco_loss", taco_loss, iteration)
            self.add_scalar("forwardtaco/training.m1_loss", m1_loss, iteration)
            self.add_scalar("forwardtaco/training.m2_loss", m2_loss, iteration)
            self.add_scalar("forwardtaco/training.dur_loss", dur_loss, iteration)
            #self.add_scalar("forwardtaco/training.mi_loss", mi_loss, iteration)
            self.add_scalar("forwardtaco/grad.norm", grad_norm, iteration)
            #self.add_scalar("forwardtaco/grad.gaf", gaf, iteration)
            self.add_scalar("forwardtaco/learning.rate", learning_rate, iteration)
            self.add_scalar("forwardtaco/duration", duration, iteration)

            if iteration > self.last_slack_log + slack_log_interval:
                self.last_slack_log = iteration
                slack_logger(f"Iteration {iteration} | sec/it {duration} | loss {total_loss}")


    def log_validation(self, reduced_loss, reduced_dur_loss, model, y, y_pred, iteration):
        self.add_scalar("forwardtaco/validation.loss", reduced_loss, iteration)
        self.add_scalar("forwardtaco/validation.dur_loss", reduced_dur_loss, iteration)
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

    def log_training(self, loginfo, total_loss, taco_loss, mi_loss, grad_norm,
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
        #m1, m2, dur_hat = model_outputs
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

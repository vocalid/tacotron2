import os
import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    mask = get_mask_from_lengths(lengths).float()
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.float() * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    dropped_mels = (mels * (1.0 - drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset, experiment, hparams, split="|"):
    preprocessing_style = hparams.preprocessing_type
    filename = None
    if preprocessing_style == "nvidia":
        if dataset == "train":
            filename = hparams.training_files
        elif dataset == "valid":
            filename = hparams.validation_files
    elif preprocessing_style == "vocalid":
        filename = os.path.join(experiment.paths["acoustic_features"], dataset + ".txt")
    if filename is None:
        raise ValueError(f"Invalid combination {preprocessing_style}/{dataset}")

    with open(filename, encoding='utf-8') as f:
        if preprocessing_style == "nvidia":
            filepaths_and_text = [line.strip().split(split) for line in f]
        elif preprocessing_style == "vocalid":
            metadata = [line.strip().split(split) for line in f]
            filepaths_and_text = [(
                os.path.join(os.path.join(experiment.paths["acoustic_features"], "mel", m[1]))
                            , m[4]) for m in metadata] 
        else:
            raise ValueError(f"Invalid preprocessing style {preprocessing_style}")
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

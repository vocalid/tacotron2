import os
import time
import argparse
import math
from pathlib import Path
import itertools
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import gradient_adaptive_factor
from model import Tacotron2, ForwardTacotron, DurationTacotron2
from loss_function import Tacotron2Loss, ForwardTacotronLoss
from logger import Tacotron2Logger, ForwardTacotronLogger
from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from utils import to_gpu


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(experiment, hparams, requires_durations):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader("train", experiment, hparams, requires_durations)
    valset = TextMelLoader("valid", experiment, hparams, requires_durations)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, trainset, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank, model_type):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        if model_type == "forwardtacotron":
            logger = ForwardTacotronLogger(os.path.join(output_directory, log_directory))
        else:
            logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams, device="cuda"):
    if hparams.model_type == "tacotron2":
        model = Tacotron2(hparams).to(device)
        model.requires_durations = False
    elif hparams.model_type == "forwardtacotron":
        model = ForwardTacotron(num_chars=hparams.n_symbols, n_mels=hparams.n_mel_channels).to(device)
        model.requires_durations = True
    elif hparams.model_type == "durationtacotron2":
        model = DurationTacotron2().to(device)
        model.requires_durations = True

    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            mel_lens = x[4]
            dur =  x[7]
            y_pred = model(x)
            loss = criterion(y_pred, y, mel_lens, dur)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def calculate_global_mean(data_loader, global_mean_npy):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return to_gpu(torch.tensor(global_mean))
    sums = []
    frames = []
    print('calculating global mean...')
    for i, batch in enumerate(data_loader):
        (text_padded, input_lengths, mel_padded, gate_padded,
         output_lengths, ctc_text, ctc_text_lengths, ids) = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
    global_mean = sum(sums) / sum(frames)
    global_mean = to_gpu(global_mean.float())
    if global_mean_npy:
        np.save(global_mean_npy, global_mean.cpu().numpy())
    return global_mean


def create_align_features(attn, mel_lens, text_input_lens, ids, dur_path):
    attn = attn.data.cpu().numpy()
    bs, chars = attn.shape[0], attn.shape[2]
    argmax = np.argmax(attn[:, :, :], axis=2)
    mel_counts = np.zeros(shape=(bs, chars), dtype=np.int32)
    for b in range(bs):
        # fix random jumps in attention 
        # go along the inputs
        for j in range(1, argmax.shape[1]):
            # difference between the "center"
            if abs(argmax[b, j] - argmax[b, j-1]) > 10:
                argmax[b, j] = argmax[b, j-1]
        count = np.bincount(argmax[b, :mel_lens[b]])
        mel_counts[b, :len(count)] = count[:len(count)]

    for j, item_id in enumerate(ids):
        dur_file = os.path.join(dur_path, f'{item_id}.npy')
        #np.save(dur_file, np.trim_zeros(mel_counts[j, :], 'b'), allow_pickle=False)
        np.save(dur_file, mel_counts[j, :text_input_lens[j]], allow_pickle=False)
        #print(f"Saving durs for {item_id}")
        #print(f"argmax: {argmax[j]}")
        #print(f"Saving mel_counts: {np.trim_zeros(mel_counts[j, :], 'b')}")
        #print(f"Saving mel_lens: {mel_lens[j]}")


def create_gta_features(experiment, model,
                        train_set: DataLoader,
                        val_set: DataLoader):
    num_samples = 0
    feat_path = experiment.paths["acoustic2wavegen_training_features"]
    gta_path = os.path.join(feat_path, "gta")
    os.makedirs(gta_path, exist_ok=True)
    dur_path = os.path.join(experiment.paths["acoustic_features"], "dur")
    os.makedirs(dur_path, exist_ok=True)
    map_file = os.path.join(feat_path, "map.txt")
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    #dataloader gets text_padded, input_lengths, mel_padded, gate_padded, \
    #  output_lengths, ctc_text_paded, ctc_text_lengths, filenames
    with open(map_file, "wt") as map_file_fp:
        for i, batch in enumerate(dataset, 1):
            (text_input, text_input_lens, mels, _, mel_lens, _, _, ids, dur) = batch
            # [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
            #_, mel_out, mel_out_postnet, gate_out, alignments = model_output
            #x, mels = x.to(device), mels.to(device)
            with torch.no_grad():
                #_, gta, _ = model(x, mels)
                #_, mel_out, mel_out_postnet, _, alignment = model(x, mels)
                x, y = model.parse_batch(batch)
                y_pred = model(x)
            #gta = gta.cpu().numpy()
            _, mel_out, mel_out_postnet, gate_out, alignments = y_pred
            create_align_features(alignments, mel_lens, text_input_lens,  ids, dur_path)
            gta = mel_out_postnet.cpu().numpy()
            # iterate over items in batch
            for j, item_id in enumerate(ids):
                mel = gta[j][:, :mel_lens[j]]
                # mel = (mel + 4) / 8 TODO?
                gta_file = os.path.join(gta_path, f'{item_id}.npy')
                audio_file = os.path.join(experiment.paths["acoustic_features"], "wav", f"audio-{item_id}.npy")
                mel_file = os.path.join(experiment.paths["acoustic_features"], "mel", f"mel-{item_id}.npy")
                np.save(gta_file, mel.T, allow_pickle=False)
                # audiopath|melgtpath|melgtapath|nothing|transcript
                map_file_fp.write(f"{audio_file}|{mel_file}|{gta_file}|<no_g>|\n")
                num_samples += 1
            msg = f'{i}/{iters} Batches '
            print(msg)
        print(f"Wrote {num_samples} GTA + dur samples")


def train(experiment, output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, max_steps=150000):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): hparams object containing configuration.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # create model - does not load weights yet
    model = load_model(hparams)

    global_mean_path = os.path.join(experiment.paths["acoustic_features"], "global_mean.npy")
    train_loader, trainset, valset, collate_fn = prepare_dataloaders(experiment, hparams, model.requires_durations)
    if hparams.drop_frame_rate > 0.:
        global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy)
        hparams.global_mean = global_mean

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    if hparams.model_type == "forwardtacotron":
        print("Using ForwardTacotronLoss")
        criterion = ForwardTacotronLoss()
    else:
        print("Using TacotronLoss")
        criterion = Tacotron2Loss()


    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank, hparams.model_type)


    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    #for epoch in range(epoch_offset, hparams.epochs):
    epoch = epoch_offset
    while iteration < max_steps:
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            mel_lens = x[4]
            dur =  x[7]
            y_pred = model(x)

            loss, loginfo = criterion(y_pred, y, mel_lens, dur)
            if model.mi is not None:
                # transpose to [b, T, dim]
                decoder_outputs = y_pred[0].transpose(2, 1)
                ctc_text, ctc_text_lengths, aco_lengths = x[-2], x[-1], x[4]
                taco_loss = loss
                mi_loss = model.mi(decoder_outputs, ctc_text, aco_lengths, ctc_text_lengths)
                if hparams.use_gaf:
                    if i % gradient_adaptive_factor.UPDATE_GAF_EVERY_N_STEP == 0:
                        safe_loss = 0. * sum([x.sum() for x in model.parameters()])
                        gaf = gradient_adaptive_factor.calc_grad_adapt_factor(
                            taco_loss + safe_loss, mi_loss + safe_loss, model.parameters(), optimizer)
                        gaf = min(gaf, hparams.max_gaf)
                else:
                    gaf = 1.0
                loss = loss + gaf * mi_loss
            else:
                taco_loss = loss
                mi_loss = torch.tensor([-1.0])
                gaf = -1.0
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                taco_loss = reduce_tensor(taco_loss.data, n_gpus).item()
                mi_loss = reduce_tensor(mi_loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
                taco_loss = taco_loss.item()
                mi_loss = mi_loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.4f} mi_loss {:.4f} Grad Norm {:.4f} "
                      "gaf {:.4f} {:.2f}s/it".format(
                    iteration, taco_loss, mi_loss, grad_norm, gaf, duration))
                logger.log_training(
                    loginfo, reduced_loss, taco_loss, mi_loss, grad_norm, gaf,
                    learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    best_checkpoint_path = os.path.join(
                        output_directory, "checkpoint_best".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    best_checkpoint_path)

            iteration += 1
        epoch += 1

    # generate GTA features and leave
    train_loader_tmp = DataLoader(trainset, num_workers=0, shuffle=False,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(valset, num_workers=0,
                            shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=False, collate_fn=collate_fn, drop_last=False)
    create_gta_features(experiment, model, train_loader_tmp, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

import dsp.audio as audio
from encoder import inference as encoder
import tacorn.embeddings as embeddings
from tacorn.embeddings import extract_speaker_id


def average_embeddings(embeds, hparams):
    avg_embeds = {}
    avg_embed_counts = {}
    for path, embed in embeds:
        sid = extract_speaker_id(path, hparams)
        if sid not in avg_embeds:
            avg_embeds[sid] = embed
        else:
            avg_embeds[sid] += embed
        if sid not in avg_embed_counts:
            avg_embed_counts[sid] = 0
        else:
            avg_embed_counts[sid] += 1
    for sid in avg_embeds:
        avg_embeds[sid] /= avg_embed_counts[sid]
    return avg_embeds


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)
    return str(wav_fpath), embed


def create_embeddings(metadata, synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int, hparams):
    wav_dir = synthesizer_root.joinpath("audio")
    assert wav_dir.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_index_path = synthesizer_root.joinpath("speaker_embeddings.json")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[6])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    embeds = list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
    avg_embeds = average_embeddings(embeds, hparams)
    embeddings.save_speaker_embeddings(embed_index_path, avg_embeds, default_speaker=list(avg_embeds.keys())[0],
                                       append=True)


def write_metadata(metadata, out_dir, hparams):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hparams.sampling_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


def preprocess(experiment, hparams, wav_dir, metadata_file, n_jobs=4, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset for a given experiment

    Args:
        experiment: the experiment to run preprocessing on
        hparams: hyper parameters to use for preprocessing
        wav_dir: Path the the input wav files
        metadata_file: the metadata file containing the transcriptions
        n_jobs: Optional, number of worker process to parallelize across
        tqdm: Optional, provides a nice progress bar

    Returns:
        A list of tuple describing the train examples. this should be written to train.txt
    """
    out_dir = experiment.paths["acoustic_features"]
    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    mel_out_dir = os.path.join(out_dir, "mel")
    os.makedirs(mel_out_dir, exist_ok=True)
    wav_out_dir = os.path.join(out_dir, "wav")
    os.makedirs(wav_out_dir, exist_ok=True)
    
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    with open(metadata_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')

            if len(parts) != 3 or not parts[0] or not parts[2]:
                continue
            basename = parts[0]
            wav_path = os.path.join(wav_dir, '{}.wav'.format(basename))
            text = parts[2]
            futures.append(executor.submit(
                partial(_process_utterance, wav_path, text, mel_out_dir, wav_out_dir, basename, hparams)))
            index += 1

    metadata = [future.result() for future in tqdm(futures) if future.result() is not None]
    if hparams.speaker_embeddings:
        print("Generating speaker embeddings")
        create_embeddings(metadata, Path(out_dir), Path("encoder/saved_models/pretrained.pt"), 4, hparams)
    write_metadata(metadata, out_dir, hparams)


def _process_utterance(wav_path, text, mel_out_dir, wav_out_dir, index, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sampling_rate)
    except FileNotFoundError:  # catch missing wav exception
        return None

    # rescale wav
    if hparams.rescale_wav: #TODO
        wav = wav / np.abs(wav).max() * hparams.rescale_max

    if hparams.trim_silence:
        len_pre = len(wav)
        wav = audio.trim_silence(wav, hparams.sampling_rate)
        len_pruned = len(wav) - len_pre
        print("Trimmed " + str(len_pruned) + " samples from " + wav_path)

    # [-1, 1]
    # TODO
    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav,
                        preemphasis_factor=hparams.preemphasis_factor,
                        sampling_rate=hparams.sampling_rate,
                        num_mels=hparams.n_mel_channels,
                        n_fft=hparams.filter_length,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_length,
                        ref_level_db=hparams.ref_level_db,
                        min_level_db=hparams.min_level_db,
                        fmin=hparams.mel_fmin,
                        fmax=hparams.mel_fmax,
                        signal_normalization=hparams.signal_normalization,
                        allow_clipping_in_normalization=hparams.allow_clipping_in_normalization,
                        symmetric_mels=hparams.symmetric_mels,
                        max_abs_value=hparams.max_abs_value
                        ).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.drop_mels_length:
        return None

    # Compute the linear scale spectrogram from the wav
    #linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    #linear_frames = linear_spectrogram.shape[1]

    # sanity check
    #assert linear_frames == mel_frames

    # Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, hparams.filter_length, hparams.hop_length)

    # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, pad, mode='reflect')

    assert len(out) >= mel_frames * hparams.hop_length

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * hparams.hop_length]
    assert len(out) % hparams.hop_length == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = 'audio-{}.npy'.format(index)
    mel_filename = 'mel-{}.npy'.format(index)
    embed_filename = 'embed-{}.npy'.format(index)
    np.save(os.path.join(wav_out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(mel_out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    if hparams.speaker_embeddings:
        return (audio_filename, mel_filename, time_steps, mel_frames, text, embed_filename)
    else:
        return (audio_filename, mel_filename, time_steps, mel_frames, text)

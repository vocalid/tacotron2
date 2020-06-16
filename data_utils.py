import os
import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, sequence_to_ctc_sequence
from textanalysis.textanalyzer import TextAnalyzer
from tacorn.utterance import Utterance


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset, experiment, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(dataset, experiment, hparams)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.hparams = hparams
        if hparams.preprocessing_type == "vocalid":
            # vocalid preprocessing is never on the fly
            self.load_mel_from_disk = True
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        #TODO: will go to preprocessing
        self.textanalyzer = TextAnalyzer(use_phones=hparams.use_phonemes,
                                         g2p_backend=hparams.g2p_backend, language=hparams.language)
        self._phone_cache_dir = os.path.join(experiment.paths["acoustic_features"], "utt")
        self._hparams = hparams
        print(f"Creating new in-memory phone cache")
        self._phoneme_cache = {}
        os.makedirs(self._phone_cache_dir, exist_ok=True)
        # fill phoneme cache first time before multiprocessing clones this data
        for paths in self.audiopaths_and_text:
            self.get_mel_text_pair(paths, dummy_mel=True)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)


    def _load_sequence(self, cache_dir, filename):
        fn = filename.replace("audio-", "phones-").replace(".npy", "")
        try:
            print(f"Loading {cache_dir}/{fn}")
            ph_seq = np.load(os.path.join(cache_dir, fn + ".npy"))
            if len(ph_seq) > 0:
                return ph_seq
            else:
                print("Loading failed, running text analysis")
                return None
        except:
            print("Loading failed, running text analysis")
            return None

    def _save_sequence(self, cache_dir, filename, ph_seq):
        fn = filename.replace("audio-", "phones-")
        print(f"Storing into {cache_dir} {fn}")
        np.save(os.path.join(cache_dir, fn), ph_seq)

    def _save_utterance(self, cache_dir, filename, utt):
        fn = filename.replace("audio-", "utt-").replace(".npy", "") + ".txt"
        with open(os.path.join(cache_dir, fn), 'wt') as fp:
            fp.write(str(utt))

    def _text_to_sequence(self, filename, text):
        ''' Get a sequence of symbol IDs for a given text. '''
        if self._hparams.use_phonemes:
            # load from in-memory cache
            #print(f"Searching for {text} in cache (currently {len(self._phoneme_cache)} entries")
            if text in self._phoneme_cache:
                return self._phoneme_cache[text]
            else:
                # load from file
                ph_seq = self._load_sequence(self._phone_cache_dir, filename)
                if ph_seq is None:
                    # otherwise run TA
                    print(f"Analyzing: {text}")
                    utterance = Utterance(str(text))
                    self.textanalyzer.analyze(utterance)
                    ph_seq = np.asarray(utterance.symbol_sequence, dtype=np.int32)
                    # save utterance, sequence and store in in-memory cache
                    self._save_utterance(self._phone_cache_dir, filename, utterance)
                    self._save_sequence(self._phone_cache_dir, filename, ph_seq)
                self._phoneme_cache[text] = ph_seq
                return ph_seq
        utterance = Utterance(text)
        self.textanalyzer.analyze(utterance)
        self._save_utterance(self._phone_cache_dir, filename, utterance)
        return np.asarray(utterance.symbol_sequence, dtype=np.int32)

    def get_mel_text_pair(self, audiopath_and_text, dummy_mel=False):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        filename = os.path.basename(audiopath)[:-4].replace("mel-", "")
        text, ctc_text = self.get_text(filename, text)
        mel = None
        if not dummy_mel:
            mel = self.get_mel(audiopath)
        return (text, ctc_text, mel, filename)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            if self.hparams.preprocessing_type == "vocalid":
                melspec = melspec.T
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, filename, text):
        sequence = self._text_to_sequence(filename, text)
        text_norm = torch.IntTensor(sequence)
        ctc_text_norm = torch.IntTensor(sequence_to_ctc_sequence(sequence))
        return text_norm, ctc_text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        max_ctc_txt_len = max([len(x[1]) for x in batch])
        ctc_text_paded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_paded .zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][1]
            ctc_text_paded[i, :ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        filenames = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            # add filenames
            filenames.append(batch[ids_sorted_decreasing[i]][3])

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text_paded, ctc_text_lengths, filenames

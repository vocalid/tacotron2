import tensorflow as tf
from textanalysis.symbols import get_symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        use_phonemes=True,
        g2p_backend="phonemizer",  # can be "phonemizer" or "vocalid"
        language="en-us",

        ################################
        # Audio Parameters             #
        ################################
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=30.0,
        mel_fmax=8000.0,
        preprocessing_type="vocalid",  # vocalid or nvidia (for compatibility with pretrained waveglow models)

        ################################
        # Nvidia style preprocessing   #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],
        max_wav_value=32768.0,

        ################################
        # TF Taco style preprocessing  #
        ################################
        valid_samples=20,  # number of files to use for the validation set
        trim_silence=True,
        drop_mels_length=True,
        max_mel_frames=800,
        min_level_db=-100,
        ref_level_db=20,
        preemphasis_factor=0.97,
        power=1.5,  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
        griffin_lim_iters=30,  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
        signal_normalization=True,  # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
        symmetric_mels=True, # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
        max_abs_value=4.,
        rescale_wav=True,  # Whether to rescale audio prior to preprocessing
        rescale_max=0.999,  # Rescaling value

        ################################
        # Speaker embeddings           #
        ################################
        speaker_embeddings=False,            # use speaker embeddings in Tacotron
        speaker_embedding_size=256,          # dimension of embedding vector
        speaker_embedding_average=True,     # use embeddings averaged per speaker instead of per-sentence
        speaker_id_separator='_',            # separator for speaker name prefix in filenames

        ################################
        # Model Parameters             #
        ################################
        model_type="tacotron2",  # tacotron2, forwardtacotron, durationtacotron2
        positional_index=False, # in forwardtacotron add positional index when expanding states

        n_symbols=None,
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=True,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        # batch_size=64,
        batch_size=32,
        mask_padding=True,  # set model's padded outputs to padded values

        ##################################
        # MMI options                    #
        ##################################
        drop_frame_rate=0.2,
        use_mmi=False,
        use_gaf=False,
        max_gaf=0.5,
        global_mean_npy='ljspeech_global_mean.npy'
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    hparams.n_symbols = len(get_symbols(hparams.use_phonemes, hparams.g2p_backend)[0])
    return hparams

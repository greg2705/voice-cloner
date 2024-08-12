import gc
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torchaudio
from numpy.random import default_rng
from pyannote.core import Annotation
from pyannote.core import Timeline
from trainer import Trainer
from trainer import TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainerConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsAudioConfig

from .stt_pipeline import diarization_torch
from .stt_pipeline import extract_speaker_audio
from .stt_pipeline import matching_stt_dia
from .stt_pipeline import stt

torch.set_num_threads(16)


def clear_gpu_cache() -> None:
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def concatenate_audios_with_silence(audio_paths: list[str]) -> tuple[torch.Tensor, int]:
    # Set the target sample rate and audio format
    target_sample_rate = 22050
    silence_duration = 1  # 1 second of silence
    silence_waveform = torch.zeros(1, target_sample_rate * silence_duration)  # 1 second of silence

    # List to store processed audio
    processed_audios = []

    for path in audio_paths:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(path)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample the audio to the target sample rate
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

        # Append the processed waveform and silence
        processed_audios.extend([waveform, silence_waveform])

    # Concatenate all the processed waveforms
    concatenated_waveform = torch.cat(processed_audios, dim=1)

    return concatenated_waveform, target_sample_rate


def suppress_overlaps_and_blanks(
    audio_paths: list[str],
    hf_token: str,
    embedding_batch_size: int = 12,
    segmentation_batch_size: int = 12,
    num_speakers: int = -1,
) -> tuple[torch.Tensor, int, Annotation]:
    """
    Suppress overlapping and blank parts from the waveform based on diarization result.

    Parameters:
    waveform (torch.Tensor): The audio waveform as a 2D torch tensor (channels x samples).
    sample_rate (int): The sample rate of the audio.
    diarization (Annotation): The diarization result as a pyannote.core.Annotation object.

    Returns:
    torch.Tensor: The processed waveform with overlapping and blank parts suppressed.
    """
    # Ensure the diarization result is an Annotation object
    waveform, sample_rate = concatenate_audios_with_silence(audio_paths)

    dia = diarization_torch(
        "pyannote/speaker-diarization-3.1",
        waveform,
        sample_rate,
        token_hf=hf_token,
        device="cuda",
        embedding_batch_size=embedding_batch_size,
        segmentation_batch_size=segmentation_batch_size,
        num_speakers=num_speakers,
    )

    timeline_overlapping_speech = Timeline(dia.get_overlap())
    timeline_blank_speech = Timeline(dia.get_timeline()).gaps()
    timeline_to_supress = timeline_overlapping_speech.union(timeline_blank_speech)
    num_samples = waveform.shape[0]
    remaining_samples = []
    current_sample = 0

    for segment in timeline_to_supress:
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)

        # Add samples before the current segment
        if start_sample > current_sample:
            remaining_samples.append(waveform[current_sample:start_sample])

        # Move the current sample pointer to the end of the current segment
        current_sample = end_sample

    # Add any remaining samples after the last segment
    if current_sample < num_samples:
        remaining_samples.append(waveform[current_sample:])

    # Concatenate all remaining samples
    return torch.cat(remaining_samples), sample_rate, dia


def suppress_overlaps_and_blanks_app(
    audio_paths: list[str],
    hf_token: str,
    embedding_batch_size: int = 12,
    segmentation_batch_size: int = 12,
    num_speakers: int = -1,
) -> tuple[torch.Tensor, int, Annotation]:
    """
    Suppress overlapping and blank parts from the waveform based on diarization result.

    Parameters:
    waveform (torch.Tensor): The audio waveform as a 2D torch tensor (channels x samples).
    sample_rate (int): The sample rate of the audio.
    diarization (Annotation): The diarization result as a pyannote.core.Annotation object.

    Returns:
    torch.Tensor: The processed waveform with overlapping and blank parts suppressed.
    """
    # Ensure the diarization result is an Annotation object
    waveform, sample_rate = concatenate_audios_with_silence(audio_paths)

    dia = diarization_torch(
        "pyannote/speaker-diarization-3.1",
        waveform,
        sample_rate,
        token_hf=hf_token,
        device="cuda",
        embedding_batch_size=embedding_batch_size,
        segmentation_batch_size=segmentation_batch_size,
        num_speakers=num_speakers,
    )

    torchaudio.save("Upload_Temp/concatenated.wav", waveform, sample_rate)
    res_dia = extract_speaker_audio(dia, "Upload_Temp/concatenated.wav")

    timeline_overlapping_speech = Timeline(dia.get_overlap())
    timeline_blank_speech = Timeline(dia.get_timeline()).gaps()
    timeline_to_supress = timeline_overlapping_speech.union(timeline_blank_speech)
    num_samples = waveform.shape[0]
    remaining_samples = []
    current_sample = 0

    for segment in timeline_to_supress:
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)

        # Add samples before the current segment
        if start_sample > current_sample:
            remaining_samples.append(waveform[current_sample:start_sample])

        # Move the current sample pointer to the end of the current segment
        current_sample = end_sample

    # Add any remaining samples after the last segment
    if current_sample < num_samples:
        remaining_samples.append(waveform[current_sample:])

    # Concatenate all remaining samples
    return torch.cat(remaining_samples), sample_rate, res_dia


def get_matching_audio_speakers(
    waveform: torch.Tensor,
    sample_rate: int,
    lst_speakers: list[str],
    hf_token: str,
    stt_path: str,
    lang: str,
    embedding_batch_size: int = 12,
    segmentation_batch_size: int = 12,
    stt_batch_size: int = 12,
    num_speakers: int = -1,
) -> pd.DataFrame:
    torchaudio.save("temp.wav", waveform, sample_rate)
    dia = diarization_torch(
        "pyannote/speaker-diarization-3.1",
        waveform,
        sample_rate,
        token_hf=hf_token,
        device="cuda",
        embedding_batch_size=embedding_batch_size,
        segmentation_batch_size=segmentation_batch_size,
        num_speakers=num_speakers,
    )
    res_stt = stt(stt_path, "temp.wav", lang=lang, device="cuda", batch_size=stt_batch_size, compute_type="int8")
    os.remove("temp.wav")
    res = pd.DataFrame(matching_stt_dia(res_stt, dia))
    return res[res["speaker"].isin(lst_speakers)]


def target_time(max_length: int) -> float:
    mean_dev = max_length / 2
    std_dev = 3.0
    num_samples = 1
    samples = -1
    rng = default_rng()
    while samples < 0 or samples > max_length:  # Number of samples to generate
        samples = rng.normal(loc=mean_dev, scale=std_dev, size=num_samples)[0]
    return samples


def create_audio(matching_dia_stt: any, max_length: int) -> list:
    matching_dia_stt = matching_dia_stt.reset_index(drop=True)
    matching_dia_stt["next_start"] = matching_dia_stt["start"].shift(-1)
    matching_dia_stt["diff"] = matching_dia_stt["next_start"] - matching_dia_stt["end"]

    res = []
    current_start = 0
    current_text = ""
    current_target_time = target_time(max_length)

    for i in matching_dia_stt.index[:-1:]:
        if current_start == 0:
            current_start = matching_dia_stt["start"][i]

        current_text += matching_dia_stt["text"][i]
        delta = matching_dia_stt["end"][i] - current_start
        next_delta = matching_dia_stt["end"][i + 1] - current_start

        if matching_dia_stt["diff"][i] > 2:
            res.append({"start": current_start, "end": matching_dia_stt["end"][i], "text": current_text})
            current_start = 0
            current_text = ""
            current_target_time = target_time(max_length)

        if np.abs(delta - current_target_time) < 0.5 or np.abs(delta - current_target_time) < np.abs(
            next_delta - current_target_time
        ):
            res.append({"start": current_start, "end": matching_dia_stt["end"][i], "text": current_text})
            current_start = 0
            current_text = ""
            current_target_time = target_time(max_length)

    return res


def create_dataset(
    waveform: torch.tensor, sp_rate: int, path_dataset: str, matching_dia_stt: pd.DataFrame, max_length: int = 15
) -> str:
    # Initialize metadata
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    padding = 0.15 * sp_rate
    os.makedirs(path_dataset, exist_ok=True)
    wavs_path = os.path.join(path_dataset, "wavs")
    os.makedirs(wavs_path, exist_ok=True)
    wavs_counter = 1
    for speakers in matching_dia_stt["speaker"].unique():
        audio_split = create_audio(matching_dia_stt, max_length)

        for new_audio in audio_split:
            # Calculate the start and end sample indices
            start_sample = int(new_audio["start"] * sp_rate - padding)
            end_sample = int(new_audio["end"] * sp_rate + padding)
            namefile = f"audio{wavs_counter}.wav"
            segment = waveform[:, start_sample:end_sample]
            torchaudio.save(wavs_path + "/" + namefile, segment, sp_rate)
            metadata["audio_file"].append("wavs/" + namefile)
            metadata["text"].append(new_audio["text"])
            metadata["speaker_name"].append(speakers)
            wavs_counter += 1

    metadata_df = pd.DataFrame(metadata)
    metadata_df = metadata_df[metadata_df["text"].str.len() >= 10].reset_index(drop=True)

    train_df = metadata_df.sample(frac=1 - 0.15).reset_index(drop=True)
    eval_df = metadata_df.drop(train_df.index).reset_index(drop=True)

    # Save to CSV
    train_metadata_path = os.path.join(path_dataset, "metadata_train.csv")
    eval_metadata_path = os.path.join(path_dataset, "metadata_eval.csv")

    train_df.to_csv(train_metadata_path, sep="|", index=False)
    eval_df.to_csv(eval_metadata_path, sep="|", index=False)

    return path_dataset


def create_model_args(model_path: str, max_length: int, sp_rate: int) -> GPTArgs:
    return GPTArgs(
        max_conditioning_length=max_length
        * sp_rate,  # the audio you will use for conditioning latents should be less than this
        min_conditioning_length=sp_rate,  # and more than this
        debug_loading_failures=True,  # this will print output to console and help you find problems in your ds
        max_wav_length=max_length * sp_rate,  # set this to >= the longest audio in your dataset
        max_text_length=200,
        mel_norm_file=model_path + "/mel_stats.pth",
        dvae_checkpoint=model_path + "/dvae.pth",
        xtts_checkpoint=model_path + "/model.pth",
        tokenizer_file=model_path + "/vocab.json",
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )


def create_audio_config(sp_rate: int) -> XttsAudioConfig:
    return XttsAudioConfig(sample_rate=sp_rate, dvae_sample_rate=sp_rate, output_sample_rate=24000)


def create_trainer_config(
    model_args: GPTArgs, audio_config: XttsAudioConfig, out_path: str, batch_size: int, nb_epochs: int
) -> GPTTrainerConfig:
    os.makedirs(out_path, exist_ok=True)
    return GPTTrainerConfig(
        run_eval=True,
        epochs=nb_epochs,
        output_path=out_path,
        model_args=model_args,
        run_name="finetuning",
        project_name="Recipe Finetuning",
        run_description="finetune",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=8,  # consider decreasing if your jupyter env is crashing or similar
        eval_split_max_size=256,
        print_step=50,
        save_best_after=0,
        save_all_best=False,
        log_model_step=1000,
        save_step=99999999,
        save_n_checkpoints=0,
        save_checkpoints=False,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )


def training(
    config: GPTTrainerConfig, lang: str, training_path: str, GRAD_ACUMM_STEPS: int, out_path: str, base_model_path: str
) -> None:
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=training_path + "/",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        language=lang,
    )
    DATASETS_CONFIG_LIST = [config_dataset]

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=out_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    del trainer, train_samples, eval_samples, model, config_dataset, DATASETS_CONFIG_LIST
    clear_gpu_cache()
    gc.collect()

    for handler in logging.getLogger("trainer").handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.getLogger("trainer").removeHandler(handler)

    subdirs = [d for d in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, d))]
    for subdir in subdirs:
        files = list(os.listdir(os.path.join(out_path, subdir)))
        if "best_model.pth" in files:
            model_src = os.path.join(out_path + "/" + subdir, "best_model.pth")
            model_dst = os.path.join(out_path, "model.pth")
            shutil.copy(model_src, model_dst)

        shutil.rmtree(out_path + "/" + subdir, ignore_errors=True)

    config_src = os.path.join(base_model_path, "config.json")
    config_dst = os.path.join(out_path, "config.json")
    shutil.copy(config_src, config_dst)

    vocab_src = os.path.join(base_model_path, "vocab.json")
    vocab_dst = os.path.join(out_path, "vocab.json")
    shutil.copy(vocab_src, vocab_dst)

    speaker_src = os.path.join(base_model_path, "speakers_xtts.pth")
    speaker_dst = os.path.join(out_path, "speakers_xtts.pth")
    shutil.copy(speaker_src, speaker_dst)

    checkpoint = torch.load(model_dst, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_dst)
    torch.save(checkpoint, model_dst)

    print("Your finetune model is avaible here : ", out_path)

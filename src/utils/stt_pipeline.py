import gc

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from faster_whisper import BatchedInferencePipeline
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydub import AudioSegment


def stt(
    model_path: str,
    audio_file: str,
    lang: str,
    device: str = "cuda",
    batch_size: int = 12,
    compute_type: str = "float16",
) -> dict:
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model, language=lang, use_vad_model=True)
    segments, _ = batched_model.transcribe(audio_file, batch_size=batch_size, word_timestamps=True, language=lang, without_timestamps=False)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    del batched_model
    return list(segments)


def stt_string(data: list) -> str:
    result = ""
    for segment in data:
        result += f"[{(segment.start):.2f}s -> {(segment.end):.2f}s] {segment.text}\n"
    return result


def diarization(model_path: str, audio_file: str, token_hf : str, device: str = "cuda", embedding_batch_size: int = 12, segmentation_batch_size: int = 12, num_speakers: int = -1) -> Annotation:

    pipeline = Pipeline.from_pretrained(model_path, use_auth_token=token_hf)
    pipeline.embedding_batch_size = embedding_batch_size
    pipeline.segmentation_batch_size = segmentation_batch_size
    pipeline.to(torch.device(device))
    waveform, sample_rate = torchaudio.load(audio_file)
    if (num_speakers > 0):
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
    else:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    gc.collect()
    torch.cuda.empty_cache()
    del pipeline
    return diarization


def diarization_torch(model_path: str, waveform: any, sample_rate : int , token_hf : str, device: str = "cuda", embedding_batch_size: int = 12, segmentation_batch_size: int = 12, num_speakers: int = -1) -> Annotation:

    pipeline = Pipeline.from_pretrained(model_path, use_auth_token=token_hf)
    pipeline.embedding_batch_size = embedding_batch_size
    pipeline.segmentation_batch_size = segmentation_batch_size
    pipeline.to(torch.device(device))

    if (num_speakers > 0):
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
    else:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    gc.collect()
    torch.cuda.empty_cache()
    del pipeline
    return diarization


def get_audio_duration(file_path: str) -> float:
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # pydub returns duration in milliseconds
    return duration_seconds


def get_speaker_durations(annotation: Annotation) -> dict[str, float]:
    """
    Takes an Annotation object and returns a dictionary with speaker labels as keys
    and their total speech duration as values.

    :param annotation: pyannote.core.Annotation object
    :return: dict with speaker labels as keys and total speech duration as values
    """
    speaker_durations = {}
    # Iterate over each unique speaker in the annotation
    for speaker in annotation.labels():
        speaker_durations[speaker] = annotation.label_duration(speaker)

    return speaker_durations


def get_speaker_times(annotation: Annotation, speaker: str) -> list[tuple[float, float]]:
    """
    Takes an Annotation object and returns a list of segments where the speaker speaks.

    :param annotation: pyannote.core.Annotation
    :param speaker: the speaker label
    :return: list of tuples representing the time intervals where the speaker speaks
    """
    timeline = annotation.label_timeline(speaker)
    speaker_times = []
    # Iterate over each segment in the timeline
    speaker_times = [(segment.start, segment.end) for segment in timeline]
    return speaker_times


def extract_speaker_audio(annotation: Annotation, audio_path: str) -> dict[str, tuple[np.ndarray, int]]:
    # Load the audio file
    y, sr = sf.read(audio_path)

    # Initialize the dictionary to store speaker audio segments
    speaker_audio = {}

    # Iterate over the segments in the annotation
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end

        # Extract the audio segment for the current speaker
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = y[start_sample:end_sample]

        # Add the audio segment to the dictionary
        if speaker not in speaker_audio:
            speaker_audio[speaker] = []
        speaker_audio[speaker].append(audio_segment)

    # Concatenate all segments for each speaker and filter by duration
    filtered_speaker_audio = {}
    for speaker in speaker_audio:
        concatenated_audio = np.concatenate(speaker_audio[speaker])
        duration = len(concatenated_audio) / sr
        if duration >= 120:
            filtered_speaker_audio[speaker] = (concatenated_audio, sr)

    return filtered_speaker_audio


def matching_stt_dia(stt_res: any , dia_res : any) -> any :
    wordbyword = [
    {"start": word.start, "end": word.end, "text": word.word}
    for seg in stt_res
    for word in seg.words]

    diarization_df = pd.DataFrame(dia_res.itertracks(yield_label=True))
    diarization_df["start"] = diarization_df[0].apply(lambda x : x.start)
    diarization_df["end"] = diarization_df[0].apply(lambda x : x.end)
    diarization_df = diarization_df.rename(columns={2: "speaker"})

    for seg in wordbyword:
        start = seg["start"]
        end = seg["end"]
        not_trivial_speakers = []

        for i in diarization_df.index :
            start_dia = diarization_df["start"][i]
            end_dia = diarization_df["end"][i]

            if (start > start_dia and end_dia > end):
                seg["speaker"] = diarization_df["speaker"][i]
                break

            not_trivial_speakers.append(np.abs(start - start_dia) + np.abs(end - end_dia))
        if ("speaker" not in seg):
            seg["speaker"] = diarization_df["speaker"][np.argmin(not_trivial_speakers)]

    return wordbyword


def stt_dia_str(res_match : any) -> str:

    speaker = ""
    lst_start = []
    lst_end = []
    lst_text = []
    lst_speaker = []

    for i in range(len(res_match)):
        if (i == 0):
            lst_start.append(res_match[i]["start"])
            speaker = res_match[i]["speaker"]
            lst_speaker.append(res_match[i]["speaker"])
            lst_text.append(res_match[i]["text"])

        elif (i == (len(res_match) - 1)):
            lst_end.append(res_match[i]["end"])
            lst_text[-1] += res_match[i]["text"]
        elif (speaker != res_match[i]["speaker"]):
            lst_end.append(res_match[i - 1]["end"])
            lst_start.append(res_match[i]["start"])
            speaker = res_match[i]["speaker"]
            lst_speaker.append(res_match[i]["speaker"])
            lst_text.append(res_match[i]["text"])

        else:
            lst_text[-1] += res_match[i]["text"]
    text = ""
    for i in range(len(lst_text)):
        text += " \n \n " + lst_speaker[i] + " " + str(round(lst_start[i], 3)) + " -> " + str(round(lst_end[i], 3)) + " : " + lst_text[i]
    return text

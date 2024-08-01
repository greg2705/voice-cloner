import contextlib
import gc
import io
import threading
import time

import numpy as np
import soundfile as sf
import torch
import whisperx
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydub import AudioSegment
from tqdm import tqdm


def stt(
    model_path: str,
    audio_file: str,
    lang: str,
    device: str = "cuda",
    batch_size: int = 12,
    compute_type: str = "float16",
) -> dict:
    model = whisperx.load_model(model_path, device, compute_type=compute_type, language=lang)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language=lang)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    return result


def diarization(model_path: str, audio_file: str, device: str = "cuda") -> Annotation:
    pipeline = Pipeline.from_pretrained(model_path)

    if device == "cuda":
        pipeline.to(torch.device("cuda"))
    result = pipeline(audio_file)
    gc.collect()
    torch.cuda.empty_cache()
    del pipeline
    return result


def get_audio_duration(file_path: str) -> float:
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # pydub returns duration in milliseconds
    return duration_seconds


def load_audio_tqdm(
    model_stt: str,
    config_path: str,
    audio_path: str,
    lang: str,
    device: str = "cuda",
    batch_size: int = 6,
    compute_type: str = "int8",
) -> tuple[Annotation, dict]:
    with contextlib.redirect_stdout(io.StringIO()):
        dia = diarization(config_path, audio_path, device=device)

        def run_pipeline() -> None:
            global result
            result = stt(model_stt, audio_path, lang, device=device, batch_size=batch_size, compute_type=compute_type)

        # Start the pipeline in the background
        pipeline_process = threading.Thread(target=run_pipeline)
        pipeline_process.start()

        for _ in tqdm(range(round(get_audio_duration(audio_path) / 34) + 6), desc="Step 3/3: Transcription", unit="s"):
            time.sleep(1)

        pipeline_process.join()

        return dia, result


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

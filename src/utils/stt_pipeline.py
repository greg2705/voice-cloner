import contextlib
import gc
import io
import threading
import time
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm


# Define the hook class
class MyGradioHookPyannote:
    def __init__(self, your_tqdm_progress_bar):
        self.your_tqdm_progress_bar = your_tqdm_progress_bar
        self.current_step_name = None

    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Optional[dict] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        # Check if the step_name has changed
        if step_name != self.current_step_name:
            # Reset the progress bar
            self.your_tqdm_progress_bar.close()
            if step_name == "embeddings":
                self.your_tqdm_progress_bar = tqdm(total=total, desc=f"Step 2/3: {step_name}")

            elif step_name == "discrete_diarization":
                self.your_tqdm_progress_bar.n = 0
                self.your_tqdm_progress_bar.refresh()
            else:
                self.your_tqdm_progress_bar = tqdm(total=total, desc=f"Step 1/3: {step_name}")
            self.current_step_name = step_name

        if total is not None and completed is not None:
            self.your_tqdm_progress_bar.total = total
            self.your_tqdm_progress_bar.update(completed - self.your_tqdm_progress_bar.n)


def stt(model_path, audio_file, lang, device="cuda", batch_size=12, compute_type="float16"):
    model = whisperx.load_model(model_path, device, compute_type=compute_type, language=lang)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language=lang)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    return result


def diarization(model_path, audio_file, device="cuda"):
    pipeline = Pipeline.from_pretrained(model_path)
    hook = MyGradioHookPyannote(tqdm(total=100))
    if device == "cuda":
        pipeline.to(torch.device("cuda"))
    result = pipeline(audio_file, hook=hook)
    gc.collect()
    torch.cuda.empty_cache()
    del pipeline
    return result


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # pydub returns duration in milliseconds
    return duration_seconds


def load_audio_tqdm(model_stt, config_path, audio_path, lang, device="cuda", batch_size=6, compute_type="int8"):
    with contextlib.redirect_stdout(io.StringIO()):
        dia = diarization(config_path, audio_path, device=device)

        def run_pipeline():
            global result
            result = stt(model_stt, audio_path, lang, device=device, batch_size=batch_size, compute_type=compute_type)

        # Start the pipeline in the background
        pipeline_process = threading.Thread(target=run_pipeline)
        pipeline_process.start()

        for _ in tqdm(range(round(get_audio_duration(audio_path) / 34) + 6), desc="Step 3/3: Transcription", unit="s"):
            time.sleep(1)

        pipeline_process.join()

        return dia, result


def get_speaker_durations(annotation):
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


def get_speaker_times(annotation, speaker):
    """
    Takes an Annotation object and returns a list of segments where the speaker speaks.

    :param timeline: pyannote.core.Annotation
    :return: list of Segment objects representing the time intervals where the speaker speaks
    """
    timeline = annotation.label_timeline(speaker)
    speaker_times = []
    # Iterate over each segment in the timeline
    for segment in timeline:
        speaker_times.append((segment.start, segment.end))
    return speaker_times


def extract_speaker_audio(annotation, audio_path):
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

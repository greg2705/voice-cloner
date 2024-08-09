import os
import shutil
from typing import Union

from huggingface_hub import snapshot_download
from pydub import AudioSegment
from pytubefix import YouTube


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created")


def download_model_hf(model_path: str, url: str, safe: bool = False) -> bool:
    if os.path.exists(model_path):
        print("Model already download")
        return True
    try:
        os.mkdir(model_path)
        print(f"Creating a directory here {model_path} and downloading the model")
        if safe:
            snapshot_download(
                repo_id=url,
                allow_patterns=["*.pth", "*.json", "*.safetensors", "*.md5", "*.py", "*.yaml"],
                local_dir=model_path,
            )
        else:
            snapshot_download(
                repo_id=url,
                allow_patterns=["*.pth", "*.json", "*.bin", "*.md5", "*.py", "*.yaml"],
                local_dir=model_path,
            )
        print("\nDownload Sucess")
    except Exception:
        return False
    return True


def clear_upload_temp() -> None:
    directory = "upload_temp"
    if not os.path.exists(directory):
        return
    shutil.rmtree(directory)
    return


def list_files_without_extension(directory: str) -> list:
    # List all files in the directory without their extensions
    return [
        os.path.splitext(file)[0] for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))
    ]


def generate_unique_filename(directory: str, filename: str) -> str:
    existing_files = list_files_without_extension(directory)
    # Generate a unique filename by appending a numerical suffix if necessary
    if filename not in existing_files:
        return filename
    counter = 1
    new_filename = f"{filename}_{counter}"
    while new_filename in existing_files:
        counter += 1
        new_filename = f"{filename}_{counter}"
    return new_filename


def get_audio_from_video(url: str, output_path: str, filename: str) -> Union[str, bool]:
    max_tries = 10
    current_try = 0
    audio_video = None
    video = YouTube(url)
    audio_video = video.streams.filter(only_audio=True).first()

    try:
        while current_try < max_tries:
            pth = audio_video.download(
                output_path=output_path, filename=filename + "." + audio_video.mime_type.split("/")[-1]
            )
            return pth
    except Exception:
        current_try += 1
        if current_try == max_tries:
            return False


def convert_to_wav_and_delete_og(input_file: str, filename: str) -> None:
    # Determine the file extension
    file_extension = os.path.splitext(input_file)[1]

    # Load the audio file
    audio = AudioSegment.from_file(input_file, format=file_extension[1:])

    # Define the output file name
    output_file = filename + ".wav"

    # Export the audio to .wav format
    audio.export(output_file, format="wav")

    # Delete the original file
    os.remove(input_file)

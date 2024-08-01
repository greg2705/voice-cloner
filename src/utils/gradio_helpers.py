# Monkey patch to solve pyutbe package

import re

from pytube import cipher


def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        # https://github.com/ytdl-org/youtube-dl/issues/29326#issuecomment-865985377
        # https://github.com/yt-dlp/yt-dlp/commit/48416bc4a8f1d5ff07d5977659cb8ece7640dcd8
        # var Bpa = [iha];
        # ...
        # a.C && (b = a.get("n")) && (b = Bpa[0](b), a.set("n", b),
        # Bpa.length || iha("")) }};
        # In the above case, `iha` is the relevant function name
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*' r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)",
        r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)",
    ]
    # logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            # logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(rf"var {re.escape(function_match.group(1))}\s*=\s*(\[.+?\]);", js)
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(  # type: ignore
        caller="get_throttling_function_name", pattern="multiple"
    )


cipher.get_throttling_function_name = get_throttling_function_name


import os
import shutil

from huggingface_hub import snapshot_download
from pydub import AudioSegment
from pytube import YouTube


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created")


def download_model_hf(model_path: str, url: str) -> bool:
    if os.path.exists(model_path):
        print("Model already download")
        return True
    try:
        os.mkdir(model_path)
        print(f"Creating a directory here {model_path} and downloading the model")
        snapshot_download(
            repo_id=url, allow_patterns=["*.pth", "*.json", "*.bin", "*.md5", "*.py", "*.yaml"], local_dir=model_path
        )
        print("Download Sucess")
    except Exception:
        return False
    return True


def clear_upload_temp() -> None:
    directory = "upload_temp"
    if os.path.exists(directory) == False:
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


def get_audio_from_video(url, output_path, filename):
    tries = 0
    for i in range(10):
        try:
            video = YouTube(url)
            audio_video = video.streams.filter(only_audio=True).first()
            pth = audio_video.download(
                output_path=output_path, filename=filename + audio_video.mime_type.split("/")[-1]
            )
            return pth

        except Exception:
            tries += 1
    return False


def convert_to_wav_and_delete_og(input_file, filename):
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

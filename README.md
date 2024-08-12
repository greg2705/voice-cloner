# Voice Cloner

Welcome to the Voice Cloner repository! This project provides easy-to-use solutions for text-to-speech (TTS), speech-to-text (STT), speaker diarization, and finetuning a XTTS model.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [I. NoUI](#i-no-ui)
  - [II. WebApp Inference](#ii-webapp-inference)
  - [III. WebApp Finetune (GPU only)](#iii-webapp-finetune)
- [Improvements](#Improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introdcution
This repository contains code and utilities for handling various speech processing tasks such as text-to-speech (TTS), speech-to-text (STT), and speaker diarization for GPU / CPU users. It can be use with a intuitive UI (check App_Inference & App_Finetuning) and without UI using the notebook recipe.ipynb.
## Features
- Speech to text using Faster Whisper batched inference pipeline.
- Speaker diarization using Pyannote.Audio
- Text to speech inference using COQUI XTTS models
- Dataset Creation from audio / youtube link in a training format
- Finetuning XTTS_Model
- Intuitive UI
  
## Requirements
- python = >=3.10,<3.12
- Poetry (https://python-poetry.org/)
-  CUDA 11.8 (Optional for finetuning and GPU accelaration)
  
## Installation
```bash
git clone https://github.com/greg2705/voice-cloner.git
cd voice-cloner
poetry lock
poetry install
```
## Usage
### I. No UI
Use the notebook recipe.ipynb (make sure to have jupyter installed, create a ipykernel and choose the right kernel). You can also find the code in utils folder.
### II. WebApp Inference
```bash
python App_Inference
```
### III. WebApp Finetune (GPU only)
```bash
python App_Finetuning
```

## Improvements
- **Inference & Cloning** : Use SOTA model, preprocess the audio to improve quality and find best parameters
- **Speech-to-text** : Use SOTA model, preprocess the audio to improve quality and use speech alignment like WhisperX to have better timestamps.
- **Diarization** : Use numbers of speakers_parameters, use SOTA model (pyannote). You can also implement the min_number_speaker and max number_speakers if you don't know the exact number.
- **Dataset Creation** : Better audio chunks splitting, completes sentences, better audio length distribution. Do dataset analysis and clean the outlier in the dataset (https://github.com/coqui-ai/TTS/tree/dev/notebooks/dataset_analysis).
- **Finetuning** : Try differents parameters , plot the different losses to analyze the learning curve, optimize memory.
  
## Contributing
The code is not perfect and may contains bugs, don't hesitate to open a PR or a Issues.
## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
## Acknowledgements
Without these project Voice Cloner would have been impossible
- https://github.com/coqui-ai/TTS
- https://github.com/SYSTRAN/faster-whispe
- https://github.com/pyannote/pyannote-audio
- https://github.com/daswer123/xtts-finetune-webui

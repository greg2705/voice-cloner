from typing import Any, tuple

import torch
import torchaudio


def load_speaker(speaker_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    loaded_speaker = torch.load(speaker_path)
    gpt_cond_latent = loaded_speaker["gpt_cond_latent"]
    speaker_embedding = loaded_speaker["speaker_embedding"]
    return gpt_cond_latent, speaker_embedding


def save_speaker(model: Any, path_audio: str, output_path: str) -> None:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=path_audio)
    torch.save({"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding}, output_path)


def generation(
    model: Any,
    text: str,
    language: str,
    gpt_cond_latent: torch.Tensor,
    speaker_embedding: torch.Tensor,
    output_path: str,
    temperature: float = 0.65,
    length_penalty: float = 1.0,
    repetition_penalty: float = 2.0,
    top_k: int = 50,
    top_p: float = 0.8,
    speed: float = 1.0,
    enable_text_splitting: bool = True,
) -> str:
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        speed=speed,
        enable_text_splitting=enable_text_splitting,
    )
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    return output_path


def one_shot_generation(
    model: Any,
    audio_path: str,
    text: str,
    language: str,
    output_path: str,
    speed: float,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
) -> str:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=audio_path)
    output_path = generation(
        model,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        output_path,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        speed=speed,
    )
    return output_path


def direct_generation(
    model: Any,
    text: str,
    language: str,
    speaker_name: str,
    output_path: str,
    speed: float,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
) -> str:
    gpt_cond_latent, speaker_embedding = load_speaker(speaker_name)
    output_path = generation(
        model,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        output_path,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        speed=speed,
    )
    return output_path

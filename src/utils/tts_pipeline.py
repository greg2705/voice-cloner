import os
import torch
import torchaudio



def load_speaker(speaker_path):
    loaded_speaker = torch.load(speaker_path)
    gpt_cond_latent = loaded_speaker['gpt_cond_latent']
    speaker_embedding = loaded_speaker['speaker_embedding']
    return gpt_cond_latent, speaker_embedding


def save_speaker(model,path_audio,output_path):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=path_audio)
    torch.save({'gpt_cond_latent': gpt_cond_latent, 'speaker_embedding': speaker_embedding},output_path)
    return

def generation(model,text,language,gpt_cond_latent,speaker_embedding ,output_path,
                      temperature=0.65,length_penalty=1.0,repetition_penalty=2.0,top_k=50,top_p=0.8,speed=1.0,enable_text_splitting=True):

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
    enable_text_splitting=enable_text_splitting)
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    return output_path

def one_shot_generation(model,audio_path,text,language,output_path,speed,temperature,repetition_penalty,top_k):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=audio_path)
    output_path=generation(model,text,language,gpt_cond_latent,speaker_embedding ,output_path,
                           temperature=temperature,repetition_penalty=repetition_penalty,top_k=top_k,speed=speed)
    return output_path

def direct_generation(model,text,language,speaker_name,output_path,speed,temperature,repetition_penalty,top_k):
    gpt_cond_latent, speaker_embedding = load_speaker(speaker_name)
    output_path=generation(model,text,language,gpt_cond_latent,speaker_embedding ,output_path,
                           temperature=temperature,repetition_penalty=repetition_penalty,top_k=top_k,speed=speed)
    return output_path
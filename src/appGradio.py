import warnings

warnings.filterwarnings("ignore")

import os
import sys

import gradio as gr
import torch
import utils.gradio_helpers
import utils.stt_pipeline
import utils.tts_pipeline
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

TRANSCRIPTION = None
DIARIZATION = None
SPEAKER_DICT = {}

# Define project path
PATH_PROJECT = os.getcwd()
print(f"Project path : {PATH_PROJECT}")

# Upload Temp path
UPLOAD_TEMP = PATH_PROJECT + "/Upload_Temp"
# Clear the upload temp when starting the app
utils.clear_upload_temp()
utils.create_directory(UPLOAD_TEMP)
os.environ["GRADIO_TEMP_DIR"] = UPLOAD_TEMP
print(f"Upload path : {UPLOAD_TEMP}")

# Speakers path
SPEAKER_PATH = PATH_PROJECT + r"\Speaker"
utils.create_directory(SPEAKER_PATH)
print(f"Speaker path : {SPEAKER_PATH}")

# Model path
MODEL_PATH = PATH_PROJECT + r"\Model"
utils.create_directory(MODEL_PATH)
print(f"Model path : {MODEL_PATH}")

# Download & Load Text-to-Speech Model
bool_dl_t2s = utils.download_model_hf(MODEL_PATH + r"\XTTS_V2", "coqui/XTTS-v2")
if bool_dl_t2s == False:
    print("Cannot download model")
    sys.exit(0)


config = XttsConfig()
config.load_json(MODEL_PATH + r"\XTTS_v2\config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH + r"\XTTS_v2", eval=True)

print("Model Text-to-Speech loaded")

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

model.to(device)

# Map Languages
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}

#  Force DarkMode
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def oneshot_tts(
    input_audio_onehsot, language_gr_oneshot, text_input_oneshot, speed_os, temperature_os, rp_penalty_os, top_k_os
):
    if input_audio_onehsot == None:
        gr.Warning("You have to upload or record a audio")
        return None

    if text_input_oneshot == "":
        gr.Warning("You have to define the text to generate")
        return None

    language = languages[language_gr_oneshot]

    output_path = UPLOAD_TEMP + "/" + utils.generate_unique_filename(UPLOAD_TEMP, "audio_synthesized") + ".wav"

    if os.path.exists(output_path):
        os.remove(output_path)
    audio = tts_pipeline.one_shot_generation(
        model,
        input_audio_onehsot,
        text_input_oneshot,
        language,
        output_path,
        float(speed_os),
        float(temperature_os),
        float(rp_penalty_os),
        int(top_k_os),
    )
    return audio


def oneshot_save(input_audio_onehsot, speaker_input_oneshot):
    if input_audio_onehsot == None:
        gr.Warning("You have to upload or record a audio")
        return
    if speaker_input_oneshot == "":
        gr.Warning("You have to set the Speaker Name")
        return

    filename = utils.generate_unique_filename(SPEAKER_PATH, speaker_input_oneshot) + ".pth"
    tts_pipeline.save_speaker(model, input_audio_onehsot, SPEAKER_PATH + "/" + filename)
    return


def generate_tts(
    language_gr_direct,
    text_input_direct,
    speaker_direct,
    speed_direct,
    temperature_direct,
    rp_penalty_direct,
    top_k_direct,
):
    if text_input_direct == "":
        gr.Warning("You have to define the text to generate")
        return None

    language = languages[language_gr_direct]

    output_path = UPLOAD_TEMP + "/" + utils.generate_unique_filename(UPLOAD_TEMP, "audio_synthesized") + ".wav"

    speaker_name = SPEAKER_PATH + "/" + speaker_direct + ".pth"
    audio = tts_pipeline.direct_generation(
        model,
        text_input_direct,
        language,
        speaker_name,
        output_path,
        float(speed_direct),
        float(temperature_direct),
        float(rp_penalty_direct),
        int(top_k_direct),
    )
    return audio


def refresh_speaker():
    return gr.Dropdown(
        choices=utils.list_files_without_extension(SPEAKER_PATH),
        value=utils.list_files_without_extension(SPEAKER_PATH)[0],
        label="Choose a speaker",
        multiselect=False,
        interactive=True,
    )


def type_file_change(type_file):
    if type_file == "Upload":
        return gr.Files(
            file_types=[".wav", ".mp3", ".flac"], label="Audio for the deep cloning", visible=True, interactive=True
        ), gr.Textbox(value="", label="Youtube links", interactive=True, visible=False)
    return gr.Files(
        file_types=[".wav", ".mp3", ".flac"], label="Audio for the deep cloning", visible=False, interactive=True
    ), gr.Textbox(value="", label="Youtube links", interactive=True, visible=True)


def load_data_ft(language_gr_deep, type_file, files_deep, youtube_links, progress=gr.Progress(track_tqdm=True)):
    global DIARIZATION
    global TRANSCRIPTION
    global SPEAKER_DICT

    if type_file == "Upload":
        audio_path = files_deep.name
    else:
        audio_path = utils.get_audio_from_video(youtube_links, "Upload_temp/", "ytb_video")

    new_audio_path = UPLOAD_TEMP + "/finetuning_file.wav"
    utils.convert_to_wav_and_delete_og(audio_path, UPLOAD_TEMP + "/finetuning_file")
    DIARIZATION, TRANSCRIPTION = stt_pipeline.load_audio_tqdm(
        MODEL_PATH + "/faster_whisper_v3",
        MODEL_PATH + "/models/config.yaml",
        new_audio_path,
        languages[language_gr_deep],
    )
    SPEAKER_DICT = stt_pipeline.extract_speaker_audio(DIARIZATION, new_audio_path)
    return gr.Label(label="Progress Load Audio:", value="Audio Loaded")


css = """
footer {visibility: hidden}
"""

# Gradio Code

with gr.Blocks(css=css, js=js_func, title="VoiceCloner", theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <p style='font-size:40px;text-align:center;margin-bottom:20px;'>
        <b>Voice Cloner</b>
        </p>
        """
    )

    with gr.Tabs():
        with gr.TabItem("Informations"):
            gr.Markdown(""" TODO !""")

        with gr.TabItem("Simple Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    input_audio_onehsot = gr.Audio(
                        sources=["upload", "microphone"],
                        interactive=True,
                        type="filepath",
                        label="Reference Audio",
                        format="wav",
                        editable=True,
                        show_download_button=True,
                    )
                    language_gr_oneshot = gr.Dropdown(
                        label="Choose language",
                        choices=[
                            "English",
                            "Spanish",
                            "French",
                            "German",
                            "Italian",
                            "Portuguese",
                            "Polish",
                            "Turkish",
                            "Russian",
                            "Dutch",
                            " Czech",
                            "Arabic",
                            "Chinese",
                            "Japanese",
                            "Korean",
                            "Hungarian",
                            "Hindi",
                        ],
                        multiselect=False,
                        value="English",
                        interactive=True,
                    )

                    text_input_oneshot = gr.Textbox(value="", label="Text to generate", interactive=True)
                    speaker_input_oneshot = gr.Textbox(value="", label="Speaker Name", interactive=True)

                    with gr.Row():
                        speed_os = gr.Slider(
                            label="Speed", minimum=0.1, maximum=2, value=1.0, step=0.1, interactive=True
                        )
                        temperature_os = gr.Slider(
                            label="Temperature", minimum=0.1, maximum=1, value=0.65, step=0.05, interactive=True
                        )

                    with gr.Row():
                        rp_penalty_os = gr.Slider(
                            label="Repetition Penalty", minimum=0.0, maximum=4.0, value=2.0, step=0.1, interactive=True
                        )
                        top_k_os = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=5, interactive=True)

                    clone_button_oneshot = gr.Button("Clone the voice", variant="primary")

                with gr.Column():
                    output_audio_oneshot = gr.Audio(
                        sources=["upload"],
                        interactive=False,
                        scale=2,
                        type="filepath",
                        label="Audio synthesized",
                        show_download_button=True,
                    )
                    dl_speaker_oneshot = gr.Button("Save the speaker", variant="primary")

            clone_button_oneshot.click(
                fn=oneshot_tts,
                inputs=[
                    input_audio_onehsot,
                    language_gr_oneshot,
                    text_input_oneshot,
                    speed_os,
                    temperature_os,
                    rp_penalty_os,
                    top_k_os,
                ],
                outputs=[output_audio_oneshot],
                queue=True,
            )

        with gr.TabItem("Deep Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    language_gr_deep = gr.Dropdown(
                        label="Choose language",
                        choices=[
                            "English",
                            "Spanish",
                            "French",
                            "German",
                            "Italian",
                            "Portuguese",
                            "Polish",
                            "Turkish",
                            "Russian",
                            "Dutch",
                            " Czech",
                            "Arabic",
                            "Chinese",
                            "Japanese",
                            "Korean",
                            "Hungarian",
                            "Hindi",
                        ],
                        multiselect=False,
                        value="English",
                        interactive=True,
                    )

                    type_file = gr.Radio(choices=["Upload", "Youtube"], value="Upload", interactive=True)
                    files_deep = gr.File(
                        file_types=[".wav", ".mp3", ".flac"],
                        label="Audio for the deep cloning",
                        visible=True,
                        interactive=True,
                    )
                    youtube_links = gr.Textbox(value="", label="Youtube links", interactive=True, visible=False)
                    progress_ld_data = gr.Label(label="Progress Load Audio:", value="No Audio")
                    button_load_data = gr.Button("Load Audio", variant="primary")

                    @gr.render(triggers=[progress_ld_data.change])
                    def render_speakers():
                        speakers = SPEAKER_DICT
                        speakers_list = []
                        with gr.Column():
                            for i in speakers.keys():
                                speakers_list.append(i)
                                with gr.Row():
                                    sp, array_audio = speakers[i][1], speakers[i][0]
                                    gr.Audio(interactive=False, label=i, scale=1, value=(sp, array_audio[: sp * 10]))
                        speakers_radio = gr.Radio(
                            choices=speakers_list, interactive=True, label="Choose speaker to deep clone"
                        )

                with gr.Column():
                    progress_ft_model = gr.Label(label="Progress Deep Cloning :", value="No Deep Cloning")
                    output_audio_deep = gr.Audio(
                        sources=["upload"],
                        interactive=False,
                        scale=2,
                        type="filepath",
                        label="Audio synthesized after deep cloning",
                        show_download_button=True,
                    )
                    dl_model_deep = gr.Button("Save the model", variant="primary")

        type_file.change(fn=type_file_change, inputs=[type_file], outputs=[files_deep, youtube_links])
        button_load_data.click(
            fn=load_data_ft,
            inputs=[language_gr_deep, type_file, files_deep, youtube_links],
            outputs=[progress_ld_data],
            queue=True,
        )

        with gr.TabItem("Direct Generation"):
            with gr.Row():
                with gr.Column():
                    language_gr_direct = gr.Dropdown(
                        label="Choose language",
                        choices=[
                            "English",
                            "Spanish",
                            "French",
                            "German",
                            "Italian",
                            "Portuguese",
                            "Polish",
                            "Turkish",
                            "Russian",
                            "Dutch",
                            " Czech",
                            "Arabic",
                            "Chinese",
                            "Japanese",
                            "Korean",
                            "Hungarian",
                            "Hindi",
                        ],
                        multiselect=False,
                        value="English",
                        interactive=True,
                    )

                    text_input_direct = gr.Textbox(value="", label="Text to generate", interactive=True)
                    speaker_direct = gr.Dropdown(
                        choices=utils.list_files_without_extension(SPEAKER_PATH),
                        value=utils.list_files_without_extension(SPEAKER_PATH)[0],
                        label="Choose a speaker",
                        multiselect=False,
                        interactive=True,
                    )
                    generate_button = gr.Button("Clone the voice", variant="primary")

                    with gr.Row():
                        speed_direct = gr.Slider(
                            label="Speed", minimum=0.1, maximum=2, value=1.0, step=0.1, interactive=True
                        )
                        temperature_direct = gr.Slider(
                            label="Temperature", minimum=0.1, maximum=1, value=0.65, step=0.05, interactive=True
                        )

                    with gr.Row():
                        rp_penalty_direct = gr.Slider(
                            label="Repetition Penalty", minimum=0.0, maximum=4.0, value=2.0, step=0.1, interactive=True
                        )
                        top_k_direct = gr.Slider(
                            label="Top K", minimum=0, maximum=100, value=50, step=5, interactive=True
                        )

                with gr.Column():
                    output_audio_direct = gr.Audio(
                        sources=["upload"],
                        interactive=False,
                        scale=2,
                        type="filepath",
                        label="Audio synthesized",
                        show_download_button=True,
                    )

            generate_button.click(
                fn=generate_tts,
                inputs=[
                    language_gr_direct,
                    text_input_direct,
                    speaker_direct,
                    speed_direct,
                    temperature_direct,
                    rp_penalty_direct,
                    top_k_direct,
                ],
                outputs=[output_audio_direct],
                queue=True,
            )
    dl_speaker_oneshot.click(fn=oneshot_save, inputs=[input_audio_onehsot, speaker_input_oneshot], queue=True).then(
        fn=refresh_speaker, outputs=[speaker_direct], queue=False
    )

print()
demo.queue()
demo.launch(allowed_paths=["."], inbrowser=True)

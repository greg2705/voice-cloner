import gc
import os
import warnings

import gradio as gr
import torch
from pyannote.audio import Pipeline
from utils.finetuning import create_audio_config
from utils.finetuning import create_dataset
from utils.finetuning import create_model_args
from utils.finetuning import create_trainer_config
from utils.finetuning import get_matching_audio_speakers
from utils.finetuning import suppress_overlaps_and_blanks_app
from utils.finetuning import training
from utils.gradio_helpers import clear_upload_temp
from utils.gradio_helpers import convert_to_wav_and_delete_og
from utils.gradio_helpers import create_directory
from utils.gradio_helpers import download_model_hf
from utils.gradio_helpers import get_audio_from_video

warnings.filterwarnings("ignore")


def main() -> None:
    print("\n")
    # Define project path
    PATH_PROJECT = os.getcwd()
    print(f"Project path : {PATH_PROJECT}")

    # Upload Temp path
    UPLOAD_TEMP = PATH_PROJECT + "/Upload_Temp"
    # Clear the upload temp when starting the app
    clear_upload_temp(UPLOAD_TEMP)
    create_directory(UPLOAD_TEMP)
    os.environ["GRADIO_TEMP_DIR"] = UPLOAD_TEMP
    print(f"Path where file will be upload : {UPLOAD_TEMP}")

    # Dataset creation project path
    DATASET_PATH = PATH_PROJECT + "/dataset"
    create_directory(DATASET_PATH)
    print(f"Dataset path : {DATASET_PATH}")

    # Finetune folder
    FINETUNE_FOLDER = PATH_PROJECT + "/finetune/"
    clear_upload_temp(FINETUNE_FOLDER)
    create_directory(FINETUNE_FOLDER)
    print(f"Finetune model path : {FINETUNE_FOLDER}")

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

    css = """
    footer {visibility: hidden}
    """

    def create_language_dropdown(default_value: str = "English") -> gr.Dropdown:
        return gr.Dropdown(
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
                "Czech",
                "Arabic",
                "Chinese",
                "Japanese",
                "Korean",
                "Hungarian",
                "Hindi",
            ],
            multiselect=False,
            value=default_value,
            interactive=True,
        )

    def download_all_model(hf_token: str) -> None:
        print("\n")
        bool = download_model_hf("xtts_v2", "coqui/XTTS-v2")
        if not bool:
            gr.Warning("Impossible to download coqui/XTTS-v2 model")
            return

        bool = download_model_hf("faster_whisper", "Systran/faster-whisper-large-v3")
        if not bool:
            gr.Warning("Impossible to download Systran/faster-whisper-large-v3")
            return

        # We do that to have the model in the cache
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        except Exception:
            gr.Warning("Impossible to download pyannote/speaker-diarization-3.1")
            return

        print("Diarization model cached")
        gc.collect()
        torch.cuda.empty_cache()
        del pipeline
        return

    def type_file_change(type_file: str) -> tuple:
        if type_file == "Upload":
            return gr.Files(
                file_types=[".wav", ".mp3", ".flac"], label="Audio for the deep cloning", visible=True, interactive=True
            ), gr.Textbox(value="", label="Youtube links", interactive=True, visible=False)
        return gr.Files(
            file_types=[".wav", ".mp3", ".flac"], label="Audio for the deep cloning", visible=False, interactive=True
        ), gr.Textbox(value="", label="Youtube links", interactive=True, visible=True)

    def load_data(
        type_file: str, files_upload: list[str], youtube_links: str, number_speakers: str, hf_token: str
    ) -> tuple:
        CONCATENATED_AUDIO = gr.State(None)
        SPEAKERS = gr.State(None)
        speakers_radio = gr.CheckboxGroup(
            choices=[], value=None, interactive=True, label="Choose speakers for creating the dataset"
        )

        if not number_speakers.isdigit():
            gr.Warning("Number of speaker needs to be a integer")
            return CONCATENATED_AUDIO, SPEAKERS, speakers_radio

        number_speakers = int(number_speakers)

        if type_file == "Upload":
            if files_upload is None:
                gr.Warning("You need to upload audios")
                return CONCATENATED_AUDIO, SPEAKERS, speakers_radio

            try:
                wave, sp_rate, res_dia = suppress_overlaps_and_blanks_app(
                    [files.name for files in files_upload], hf_token, num_speakers=number_speakers
                )
            except Exception:
                gr.Warning(
                    "Can't load audios, make sure a valid hf_token is avaible in the Information / Load Model tab and that cuda is activate"
                )
                return CONCATENATED_AUDIO, SPEAKERS, speakers_radio

            CONCATENATED_AUDIO = gr.State({"waveform": wave, "sp_rate": sp_rate})
            SPEAKERS = gr.State(res_dia)
            return (
                CONCATENATED_AUDIO,
                SPEAKERS,
                gr.CheckboxGroup(
                    choices=list(res_dia.keys()),
                    value=None,
                    interactive=True,
                    label="Choose speakers for creating the dataset",
                ),
            )

        else:
            if not youtube_links:
                gr.Warning("You need to provide a youtube link")
                return CONCATENATED_AUDIO, SPEAKERS, speakers_radio
            bool_yt = get_audio_from_video(youtube_links, UPLOAD_TEMP, "youtube_dl")

            if not bool_yt:
                gr.Warning(
                    "Enter a valid youtube link (some video can't be downloaded due to age restriction, country ...)"
                )
                return CONCATENATED_AUDIO, SPEAKERS, speakers_radio
            convert_to_wav_and_delete_og(bool_yt, "youtube_wav")

            try:
                wave, sp_rate, res_dia = suppress_overlaps_and_blanks_app(
                    ["youtube_wav.wav"], hf_token, num_speakers=number_speakers
                )
            except Exception:
                gr.Warning(
                    "Can't load youtube video, make sure a valid hf_token is avaible in the Information / Load Model tab and that cuda is activate"
                )
                return CONCATENATED_AUDIO, SPEAKERS, speakers_radio
            CONCATENATED_AUDIO = gr.State({"waveform": wave, "sp_rate": sp_rate})
            SPEAKERS = gr.State(res_dia)
            return (
                CONCATENATED_AUDIO,
                SPEAKERS,
                gr.CheckboxGroup(
                    choices=list(res_dia.keys()),
                    value=None,
                    interactive=True,
                    label="Choose speakers for creating the dataset",
                ),
            )

    def create_data(
        CONCATENATED_AUDIO: any, speakers_radio: list[str], max_audio_length: int, hf_token: str, number_speakers: str, language_gr: str
    ) -> gr.Label:
        if CONCATENATED_AUDIO is None:
            gr.Warning("Please load data")
            return gr.Label("Dataset not created")
        if speakers_radio == []:
            gr.Warning("Please select at least one speaker")
            return gr.Label("Dataset not created")

        if not number_speakers.isdigit():
            gr.Warning("Number of speaker needs to be a integer")
            return CONCATENATED_AUDIO, SPEAKERS, speakers_radio

        number_speakers = int(number_speakers)

        wave = CONCATENATED_AUDIO.value["waveform"]
        sp_rate = CONCATENATED_AUDIO.value["sp_rate"]
        lang = languages[language_gr]

        try:
            datframe_speaker = get_matching_audio_speakers(
                wave, sp_rate, speakers_radio, hf_token, "faster_whisper/", lang, num_speakers=number_speakers
            )
            create_dataset(wave, sp_rate, DATASET_PATH, datframe_speaker, max_length=max_audio_length)

        except Exception:
            gr.Warning("Please make sure hf_token is provided, models are downloaded and cuda is activate")
            return gr.Label("Dataset not created")
        print("Dataset created !")
        return gr.Label("Dataset created")

    def finetune_model_xtts(
        CONCATENATED_AUDIO: any, nb_epoch: int, batch_size: int, grad_acumm_steps: int, model_to_load: str, max_audio_length: int, language_gr: str
    ) -> gr.Label:
        dataset_ready = gr.Label("Model not finetuned")

        sp_rate = 22050 if CONCATENATED_AUDIO is None else CONCATENATED_AUDIO.value["sp_rate"]

        path_train = DATASET_PATH + "/metadata_train.csv"
        path_eval = DATASET_PATH + "/metadata_eval.csv"
        path_wavs = DATASET_PATH + "/wavs"

        if not (os.path.exists(path_train) and os.path.exists(path_eval) and os.path.exists(path_wavs)):
            gr.Warning("You have to create the dataset")
            return dataset_ready

        lang = languages[language_gr]

        try:
            sp_rate = CONCATENATED_AUDIO.value["sp_rate"]
            model_args = create_model_args(model_to_load, max_audio_length, sp_rate)
            audio_config = create_audio_config(sp_rate)
            config = create_trainer_config(model_args, audio_config, FINETUNE_FOLDER, batch_size, nb_epoch)
            training(config, lang, DATASET_PATH, grad_acumm_steps, FINETUNE_FOLDER, model_to_load)
        except Exception:
            gr.Warning(
                "Something went wrong during training, make sure you followed every steps and have enough VRAM avaible"
            )
            return dataset_ready

        return gr.Label("Model is finetuned")

    with gr.Blocks(css=css, js=js_func, title="VoiceCloner", theme=gr.themes.Soft()) as demo:
        gr.HTML("""<p style='font-size:40px;text-align:center;margin-bottom:20px;'>
            <b>Voice Cloner Finetuning</b>""")
        with gr.Tabs():
            with gr.TabItem("Information / Load Model"):
                gr.HTML("""<h2 style="font-size: 20px;">Getting Started</h2>
                    <p style="font-size: 18px">You will find a lot of informations in the terminal.</p>
<br>
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 18px;">
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol style="list-style-position: inside;">
                <strong>Download all models:</strong>
                <ul>
                    <li>Specify your HF_token(https://huggingface.co/settings/tokens)</li>
                    <li>Click the <strong>Download all Models</strong> to download models.</li>
                    <li>The model will be saved in the current folder on your system (only speaker diarization is cached).</li>
                </ul>
        </ol>
    </div>
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="2" style="list-style-position: inside;">
                <strong>Dataset Creation:</strong>
                <ul>
                    <li><strong>Choose Language:</strong> Select the desired language from a dropdown menu.</li>
                    <li><strong>Data Source:</strong> Choose between uploading audio files or using YouTube links.</li>
                    <li><strong>Number of Speakers:</strong> Specify the number of speakers for more precise outputs.</li>
                    <li><strong>Load Data:</strong> Click the Load Data button and wait to select which speaker to add to the dataset.</li>
<br><br>
                </ul>
        </ol>
</div>
<div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol style="list-style-position: inside;">
                <strong>Create Dataset:</strong>
                <ul>
                   <li><strong>Speaker Selection</strong>After data loading, choose a speakers from the provided list.</li>
                    <li><strong>Max audio length:</strong>Set the maximum length for truncated audio files.</li>
                    <li><strong>Create Dataset:</strong>Click the Create Dataset button to finalize the dataset creation.</li>
                </ul>
        </ol>
    </div>
 <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol style="list-style-position: inside;">
                <strong>Finetune the model:</strong>
                <ul>
                    <li><strong>Choose param</strong>Set the number of training epochs, define the batch size for training and specify the number of gradient accumulation steps.</li>
                    <li><strong>Fientune:</strong>Click the Finetune the model button to finetune the model.</li>
                    <li><strong>Check terminal:</strong>Check terminal to see progress on the finetuning.</li>
                </ul>
        </ol>
    </div>
</div>
    <h2 style="font-size: 2em;">Notes</h2>
        <ul style="font-size: 1.2em; line-height: 1.6;">
            <li><strong>GPU Requirements:</strong> This application requires a CUDA-compatible GPU with at least 6GB of VRAM. If you have less VRAM, modify the code to reduce batch sizes accordingly.</li>
            <li><strong>Model Saving:</strong> Before reloading the application, ensure that the finetuned model is extracted into the <code>"/finetune"</code> folder to prevent data loss.</li>
            <li><strong>Direct Finetuning:</strong> If datasets are already present in the correct format and location ("/dataset"), you can skip dataset creation and proceed directly to finetuning.</li>
        </ul>
<br><br>""")

                hf_token = gr.Textbox(value="", label="HF Token", interactive=True)
                download_model_button = gr.Button("Download all models", variant="primary")
                model_to_load = gr.Textbox(value="xtts_v2", label="Model TTS", interactive=True)

                download_model_button.click(fn=download_all_model, inputs=[hf_token], queue=True)

            with gr.TabItem("Dataset Creation"):
                CONCATENATED_AUDIO = gr.State(None)
                SPEAKERS = gr.State(None)

                with gr.Row():
                    with gr.Column():
                        language_gr = create_language_dropdown()

                        with gr.Row():
                            type_file = gr.Radio(choices=["Upload", "Youtube"], value="Upload", interactive=True)
                            number_speakers = gr.Textbox(label="Number of speaker", value="0")

                        max_audio_length = gr.Slider(
                            label="Max audio length", minimum=7, maximum=20, value=11, step=0.1, interactive=True
                        )

                        files_upload = gr.Files(
                            file_types=[".wav", ".mp3", ".flac"],
                            label="Audio for the finetuning",
                            visible=True,
                            interactive=True,
                        )
                        youtube_links = gr.Textbox(value="", label="Youtube links", interactive=True, visible=False)
                        dataset_ready = gr.Label("Dataset not created")
                        button_load_data = gr.Button(value="Load Data", variant="primary")

                        type_file.change(
                            fn=type_file_change, inputs=[type_file], outputs=[files_upload, youtube_links], queue=True
                        )

                    with gr.Column():

                        @gr.render(inputs=SPEAKERS)
                        def render_speakers(speakers: any) -> None:
                            if speakers is not None:
                                for speaker in speakers.value:
                                    sp, array_audio = speakers.value[speaker][1], speakers.value[speaker][0]
                                    gr.Audio(
                                        interactive=False, label=speaker, scale=1, value=(sp, array_audio[: sp * 10])
                                    )

                        speakers_radio = gr.CheckboxGroup(
                            choices=[], value=None, interactive=True, label="Choose speakers for creating the dataset"
                        )
                        button_create_data = gr.Button(value="Create Dataset", variant="primary")

            button_load_data.click(
                fn=load_data,
                inputs=[type_file, files_upload, youtube_links, number_speakers, hf_token],
                outputs=[CONCATENATED_AUDIO, SPEAKERS, speakers_radio],
                queue=True,
            )

            button_create_data.click(
                fn=create_data,
                inputs=[CONCATENATED_AUDIO, speakers_radio, max_audio_length, hf_token, number_speakers, language_gr],
                outputs=[dataset_ready],
                queue=True,
            )

            with gr.TabItem("Finetuning"):
                gr.HTML("")

                with gr.Row():
                    nb_epoch = gr.Slider(label="Epochs", minimum=2, maximum=200, value=15, step=1, interactive=True)
                    batch_size = gr.Slider(label="Batch size", minimum=1, maximum=30, value=3, step=1, interactive=True)
                    grad_acumm_steps = gr.Slider(
                        label="GRAD_ACUMM_STEPS", minimum=24, maximum=526, value=84, step=1, interactive=True
                    )

                dataset_ready = gr.Label("Model not finetuned")
                button_finetune = gr.Button(value="Finetune the model", variant="primary")

                button_finetune.click(
                    fn=finetune_model_xtts,
                    inputs=[
                        CONCATENATED_AUDIO,
                        nb_epoch,
                        batch_size,
                        grad_acumm_steps,
                        model_to_load,
                        max_audio_length,
                        language_gr,
                    ],
                    outputs=[dataset_ready],
                    queue=True,
                )

    demo.queue()
    demo.launch(
        allowed_paths=["."],
        inbrowser=True,
        share=False,
    )


if __name__ == "__main__":
    main()

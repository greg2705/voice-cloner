import os
import warnings

import gradio as gr
import torch
from utils.gradio_helpers import clear_upload_temp
from utils.gradio_helpers import create_directory
from utils.gradio_helpers import download_model_hf
from utils.gradio_helpers import generate_unique_filename
from utils.gradio_helpers import list_files_without_extension
from utils.tts_pipeline import direct_generation
from utils.tts_pipeline import load_model
from utils.tts_pipeline import one_shot_generation
from utils.tts_pipeline import save_speaker

warnings.filterwarnings("ignore")

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

# Speakers path
SPEAKER_PATH = PATH_PROJECT + r"\Speaker"
create_directory(SPEAKER_PATH)
print(f"Path where speaker will be save : {SPEAKER_PATH}")

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


def oneshot_tts(
    input_audio_onehsot: str,
    language_gr_oneshot: str,
    text_input_oneshot: str,
    speed_os: float,
    temperature_os: float,
    rp_penalty_os: float,
    top_k_os: int,
    MODEL: any,
) -> str:
    if MODEL is None:
        gr.Warning("You have to load a Model in order to generate audio")
        return None

    if input_audio_onehsot is None:
        gr.Warning("You have to upload or record a audio")
        return None

    if not text_input_oneshot:
        gr.Warning("You have to define the text to generate")
        return None

    language = languages[language_gr_oneshot]

    output_path = UPLOAD_TEMP + "/" + generate_unique_filename(UPLOAD_TEMP, "audio_synthesized") + ".wav"

    if os.path.exists(output_path):
        os.remove(output_path)

    audio = one_shot_generation(
        MODEL.value,
        input_audio_onehsot,
        text_input_oneshot,
        language,
        output_path,
        float(speed_os),
        float(temperature_os),
        float(rp_penalty_os),
        int(top_k_os),
    )
    print(f"Audio generated here : {output_path}")
    return audio


def generate_tts(
    language_gr_direct: str,
    text_input_direct: str,
    speaker_direct: str,
    speed_direct: float,
    temperature_direct: float,
    rp_penalty_direct: float,
    top_k_direct: int,
    MODEL: any,
) -> str:
    if MODEL is None:
        gr.Warning("You have to load a Model in order to generate audio")
        return None

    if not text_input_direct:
        gr.Warning("You have to define the text to generate")
        return None

    if speaker_direct is None:
        gr.Warning("You have to set a speaker")
        return None

    language = languages[language_gr_direct]

    output_path = UPLOAD_TEMP + "/" + generate_unique_filename(UPLOAD_TEMP, "audio_synthesized") + ".wav"

    speaker_name = SPEAKER_PATH + "/" + speaker_direct + ".pth"
    audio = direct_generation(
        MODEL.value,
        text_input_direct,
        language,
        speaker_name,
        output_path,
        float(speed_direct),
        float(temperature_direct),
        float(rp_penalty_direct),
        int(top_k_direct),
    )
    print(f"Audio generated here : {output_path}")
    return audio


def refresh_speaker() -> gr.Dropdown:
    val = list_files_without_extension(SPEAKER_PATH)
    val_0 = None if len(val) == 0 else val[0]

    return gr.Dropdown(
        choices=val,
        value=val_0,
        label="Choose a speaker",
        multiselect=False,
        interactive=True,
    )


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


def create_audio_output(label: str) -> gr.Audio:
    return gr.Audio(
        sources=["upload"],
        interactive=False,
        scale=2,
        type="filepath",
        label=label,
        show_download_button=True,
    )


def download_base_model() -> None:
    print("\n")
    bool = download_model_hf("xtts_v2", "coqui/XTTS-v2")
    if not bool:
        gr.Warning("Impossible to download base model")


def load_model_fn(model_to_load: str) -> None:
    if not os.path.exists(model_to_load):
        gr.Warning("Provide path doesn't exist")
        return "Model is not load", gr.State(None)
    try:
        model = load_model(model_to_load)
    except Exception:
        gr.Warning("Can't load the model, verify it's a XTTS model (base or finetune)")
        return "Model is not load", gr.State(None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.to(device)
        print("Using GPU")
    else:
        print("Using CPU")

    return "Model load", gr.State(model)


def oneshot_save(input_audio_onehsot: str, speaker_input_oneshot: str, MODEL: any) -> None:
    if MODEL is None:
        gr.Warning("You have to load a Model")
        return

    if input_audio_onehsot is None:
        gr.Warning("You have to upload or record a audio")
        return

    if not text_input_oneshot:
        gr.Warning("You have to set the Speaker Name")
        return

    filename = generate_unique_filename(SPEAKER_PATH, speaker_input_oneshot) + ".pth"
    save_speaker(MODEL.value, input_audio_onehsot, SPEAKER_PATH + "/" + filename)
    return


with gr.Blocks(css=css, js=js_func, title="VoiceCloner", theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <p style='font-size:40px;text-align:center;margin-bottom:20px;'>
        <b>Voice Cloner Inference</b>
        </p>
        """
    )

    with gr.Tabs():
        with gr.TabItem("Information / Load Model"):
            gr.HTML("""<h2 style="font-size: 20px;">Getting Started</h2>
                    <p style="font-size: 18px">Very important to look at the terminal.</p>
<br>
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 18px;">
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol style="list-style-position: inside;">
            <li>
                <strong>Download the Base Model:</strong>
                <ul>
                    <li>The base model, <strong>xtts_v2</strong>, developed by Coqui, is required for the application.</li>
                    <li>Click the <strong>Download Model</strong> button to download the base model.</li>
                    <li>The model will be saved in the <code>xtts_v2</code> folder on your system.</li>
                </ul>
            </li>
        </ol>
    </div>
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="2" style="list-style-position: inside;">
            <li>
                <strong>Load a Model:</strong>
                <ul>
                    <li>You can load the base model or any fine-tuned model that you have created.</li>
                    <li>Input the path to your desired <code>xtts_v2</code> model in the input field provided.</li>
                    <li>Click the <strong>Load Model</strong> button to load the model into the application.</li>
                </ul>
            </li>
        </ol>
    </div>
</div>

<h2 style="font-size: 20px;">Voice Cloning</h2>
<br>
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 18px;">
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="3" style="list-style-position: inside;">
            <li>
                <strong>Upload or Record Audio:</strong>
                <ul>
                    <li>Upload an audio file from your device or use the microphone to record new audio.</li>
                </ul>
            </li>
        </ol>
    </div>
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="4" style="list-style-position: inside;">
            <li>
                <strong>Select Language and Enter Text:</strong>
                <ul>
                    <li>Choose the language in which you want to generate text.</li>
                    <li>Enter the text that you want to convert to speech.</li>
                </ul>
            </li>
        </ol>
    </div>

    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="5" style="list-style-position: inside;">
            <br>
            <li>
                <strong>Adjust Settings:</strong>
                <ul>
                    <li>Modify the following parameters to customize the output audio:</li>
                    <ul>
                        <li><strong>Temperature:</strong> Controls the randomness of the output.</li>
                        <li><strong>Speed:</strong> Adjusts the playback speed of the generated audio.</li>
                        <li><strong>Top_k:</strong> Limits the selection of most probable tokens to k options.</li>
                        <li><strong>Repetition Penalty:</strong> Discourages repetition in the generated audio.</li>
                    </ul>
                </ul>
            </li>
        </ol>
    </div>
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <br>
        <ol start="6" style="list-style-position: inside;">
            <li>
                <strong>Generate and Save:</strong>
                <ul>
                    <li>Click on <strong>Generate Audio</strong> to create the speech output.</li>
                    <li>Download the generated audio for offline use.</li>
                    <li>Save the speaker's voice profile by assigning a name, making it available for future use in the <strong>Generation</strong> tab.</li>
                </ul>
            </li>
        </ol>
    </div>
</div>

<h2 style="font-size: 20px;">Generation</h2>
<br>
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 18px;">
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="7" style="list-style-position: inside;">
            <li>
                <strong>Select a Saved Speaker:</strong>
                <ul>
                    <li>Choose from the list of saved speaker profiles that you have previously created in the Voice Cloning tab.</li>
                </ul>
            </li>
        </ol>
    </div>
    <div style="width: 48%; text-align: left; box-sizing: border-box;">
        <ol start="8" style="list-style-position: inside;">
            <li>
                <strong>Generate Audio:</strong>
                <ul>
                    <li>Enter the text you wish to convert into speech.</li>
                    <li>Use the same customization settings to tweak the audio output as per your preference.</li>
                </ul>
            </li>
        </ol>
    </div>
</div>
<br><br>""")

            download_model_button = gr.Button("Download base model", variant="primary")
            model_to_load = gr.Textbox(value="xtts_v2", label="Path to the model", interactive=True)
            load_model_button = gr.Button(value="Load model", variant="primary")
            label_model_load = gr.Label("No model load")
            MODEL = gr.State(None)

            download_model_button.click(fn=download_base_model)
            load_model_button.click(fn=load_model_fn, inputs=[model_to_load], outputs=[label_model_load, MODEL])

        with gr.TabItem("Voice Cloning"):
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

                    language_gr_oneshot = create_language_dropdown()

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

                    text_input_oneshot = gr.Textbox(value="", label="Text to generate", interactive=True)
                    clone_button_oneshot = gr.Button("Clone the voice", variant="primary")

                with gr.Column():
                    output_audio_oneshot = create_audio_output("Audio synthesized")
                    speaker_input_oneshot = gr.Textbox(value="", label="Speaker Name", interactive=True)
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
                            MODEL,
                        ],
                        outputs=[output_audio_oneshot],
                        queue=True,
                    )

        with gr.TabItem("Generation"):
            with gr.Row():
                with gr.Column():
                    language_gr_direct = create_language_dropdown()

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

                    text_input_direct = gr.Textbox(value="", label="Text to generate", interactive=True)
                    speaker_direct = gr.Dropdown(
                        choices=list_files_without_extension(SPEAKER_PATH),
                        value=None,
                        label="Choose a speaker",
                        multiselect=False,
                        interactive=True,
                    )
                    generate_button = gr.Button("Generate", variant="primary")

                with gr.Column():
                    output_audio_direct = create_audio_output("Audio synthesized")

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
                    MODEL,
                ],
                outputs=[output_audio_direct],
                queue=True,
            )

            dl_speaker_oneshot.click(
                fn=oneshot_save, inputs=[input_audio_onehsot, speaker_input_oneshot, MODEL], queue=True
            ).then(fn=refresh_speaker, outputs=[speaker_direct], queue=False)

demo.queue()
demo.launch(allowed_paths=["."], inbrowser=True)

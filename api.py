import os
import shutil
import subprocess
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from models import voicecraft
from data.tokenizer import AudioTokenizer, TextTokenizer
from inference_tts_scale import inference_one_sample
from pydantic import BaseModel
import io
from starlette.responses import StreamingResponse
import getpass
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('api.log'),
                        logging.StreamHandler()
                    ])

app = FastAPI()

class AdditionalArgs(BaseModel):
    top_k: int = 0
    top_p: float = 0.8
    temperature: float = 1.0
    stop_repetition: int = 3
    kvcache: int = 1
    sample_batch_size: int = 4

@app.post("/generate")
async def generate_audio(
    time: float = Form(...),
    target_text: str = Form(""),
    audio: UploadFile = File(...),
    transcript: UploadFile = File(...),
    save_to_file: bool = Form(True),
    output_path: str = Form("."),
    top_k: int = Form(0),
    top_p: float = Form(0.8),
    temperature: float = Form(1.0),
    stop_repetition: int = Form(3),
    kvcache: int = Form(1),
    sample_batch_size: int = Form(4),
    device: str = Form(None)
):
    logging.info("Received request to generate audio")

    # Get the current username
    username = getpass.getuser()

    # Set the USER environment variable to the username
    os.environ['USER'] = username
    logging.debug(f"Set USER environment variable to: {username}")

    # Set the os variable for espeak
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = './espeak/libespeak-ng.dll'
    logging.debug("Set PHONEMIZER_ESPEAK_LIBRARY environment variable")

    # Create the voice folder
    voice_folder = f"./voices/{os.path.splitext(audio.filename)[0]}"
    os.makedirs(voice_folder, exist_ok=True)
    logging.debug(f"Created voice folder: {voice_folder}")

    # Save the uploaded files
    audio_fn = os.path.join(voice_folder, audio.filename)
    transcript_fn = os.path.join(voice_folder, f"{os.path.splitext(audio.filename)[0]}.txt")
    with open(audio_fn, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    with open(transcript_fn, "wb") as f:
        shutil.copyfileobj(transcript.file, f)
    logging.debug(f"Saved uploaded files: {audio_fn}, {transcript_fn}")

    # Prepare alignment if not already done
    mfa_folder = os.path.join(voice_folder, "mfa")
    os.makedirs(mfa_folder, exist_ok=True)
    alignment_file = os.path.join(mfa_folder, f"{os.path.splitext(audio.filename)[0]}.csv")
    if not os.path.isfile(alignment_file):
        logging.info("Preparing alignment...")
        subprocess.run(["mfa", "align", "-v", "--clean", "-j", "1", "--output_format", "csv",
                        voice_folder, "english_us_arpa", "english_us_arpa", mfa_folder])
        logging.info("Alignment completed")
    else:
        logging.info("Alignment file already exists. Skipping alignment.")

    # Read the alignment file and find the closest end time
    cut_off_sec = time
    prompt_end_word = ""
    closest_end = 0
    with open(alignment_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            begin, end, label, type, *_ = line.strip().split(",")
            end = float(end)
            if end > cut_off_sec:
                break
            closest_end = end
            prompt_end_word = label

    logging.info(f"Identified end value closest to desired time: {closest_end} seconds")

    if not prompt_end_word:
        logging.error("No suitable word found within the desired time frame.")
        return {"message": "No suitable word found within the desired time frame."}

    # Read the transcript file and extract the prompt
    with open(transcript_fn, "r") as f:
        transcript_text = f.read().strip()

    logging.debug(f"Reading transcript file: {transcript_fn}")

    transcript_words = transcript_text.split()
    prompt_end_idx = -1
    for idx, word in enumerate(transcript_words):
        if word.strip(".,!?;:") == prompt_end_word:
            prompt_end_idx = idx
            break

    if prompt_end_idx == -1:
        logging.error("Error: Prompt end word not found in the transcript.")
        return {"message": "Error: Prompt end word not found in the transcript."}

    prompt_transcript = " ".join(transcript_words[:prompt_end_idx+1])

    logging.info(f"Prompt transcript up to closest end word: {prompt_transcript}")

    # Prepend the extracted transcript to the user's prompt
    final_prompt = prompt_transcript + " " + target_text
    logging.info(f"Final prompt to be used: {final_prompt}")

    # Set the device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.lower() not in ["cpu", "cuda"]:
        logging.warning("Invalid device specified. Defaulting to CPU.")
        device = "cpu"

    logging.info(f"Using device: {device}")

    # Set up the model and tokenizers
    voicecraft_name = "giga330M.pth"
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    phn2num = ckpt['phn2num']
    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)
    
    additional_args = AdditionalArgs(
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        stop_repetition=stop_repetition,
        kvcache=kvcache,
        sample_batch_size=sample_batch_size
    )
    
    decode_config = {
        'top_k': additional_args.top_k,
        'top_p': additional_args.top_p,
        'temperature': additional_args.temperature,
        'stop_repetition': additional_args.stop_repetition,
        'kvcache': additional_args.kvcache,
        "codec_audio_sr": 16000,
        "codec_sr": 50,
        "silence_tokens": [1388, 1898, 131],
        "sample_batch_size": additional_args.sample_batch_size
    }

    # Calculate prompt_end_frame based on the actual closest end time
    prompt_end_frame = int(closest_end * 16000)
    logging.info(f"Prompt end frame: {prompt_end_frame}")

    logging.info("Calling inference_one_sample...")
    try:
        # Generate the audio
        concated_audio, gen_audio = inference_one_sample(
            model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer,
            audio_fn, final_prompt, device, decode_config, prompt_end_frame
        )
        logging.info("Inference completed.")
    except Exception as e:
        logging.error(f"Error occurred during inference: {str(e)}")
        return {"message": "An error occurred during audio generation."}

    if save_to_file:
        # Save the generated audio to a file
        output_file = os.path.join(output_path, f"{os.path.splitext(audio.filename)[0]}_generated.wav")
        torchaudio.save(output_file, gen_audio[0].cpu(), 16000)
        logging.info(f"Generated audio saved as: {output_file}")
        return {"message": "Audio generated successfully.", "output_file": output_file}
    else:
        # Serve the generated audio as bytes
        audio_bytes = io.BytesIO()
        torchaudio.save(audio_bytes, gen_audio[0].cpu(), 16000, format="wav")
        audio_bytes.seek(0)
        return StreamingResponse(audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8245)

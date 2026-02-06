import os
import subprocess
from pathlib import Path

# ---------------- PATHS -----------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
SADTALKER_DIR = Path("SadTalker")
WAV2LIP_DIR = Path("Wav2Lip")
VOICE_FILE = INPUT_DIR / "voice.wav"
IMAGE_FILE = INPUT_DIR / "face.jpg"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- CHECK IMAGE -----------------
if not IMAGE_FILE.exists():
    raise FileNotFoundError(f"Put your face image at {IMAGE_FILE}")

# ---------------- GENERATE VOICE -----------------
if not VOICE_FILE.exists():
    print("🎤 No voice.wav found, generating TTS (ElevenLabs)...")
    try:
        from elevenlabslib import ElevenLabsUser
        api_key = os.environ.get("ELEVENLABS_API_KEY") or input("Enter ElevenLabs API Key: ").strip()
        user = ElevenLabsUser(api_key)
        voice = user.get_voices_by_name("Rachel")
        text_prompt = input("Enter the prompt text (20-30 sec): ").strip()
        audio_bytes = voice.generate_audio_bytes(text_prompt)
        with open(VOICE_FILE, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Generated TTS at {VOICE_FILE}")
    except Exception as e:
        print("❌ Failed TTS:", e)
        exit(1)
else:
    print(f"✅ Using existing voice: {VOICE_FILE}")

# ---------------- CLONE / CHECK SADTALKER -----------------
if not SADTALKER_DIR.exists():
    print("📥 Cloning SadTalker...")
    subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"])
    subprocess.run(["bash", f"{SADTALKER_DIR}/scripts/download_models.sh"])

# ---------------- CLONE / CHECK Wav2Lip -----------------
if not WAV2LIP_DIR.exists():
    print("📥 Cloning Wav2Lip...")
    subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"])
    subprocess.run(["bash", f"{WAV2LIP_DIR}/scripts/download_weights.sh"])

# ---------------- RUN SADTALKER -----------------
print("🤖 Running SadTalker for facial animation...")
cmd_sadtalker = [
    "python3", f"{SADTALKER_DIR}/inference.py",
    "--driven_audio", str(VOICE_FILE),
    "--source_image", str(IMAGE_FILE),
    "--result_dir", str(OUTPUT_DIR),
    "--still",
    "--preprocess", "full",
    "--enhancer", "gfpgan"
]
subprocess.run(cmd_sadtalker)

# Get generated SadTalker video
generated_video = None
for file in OUTPUT_DIR.iterdir():
    if file.suffix.lower() in [".mp4", ".mov"]:
        generated_video = file
        break

if generated_video is None:
    raise FileNotFoundError("SadTalker did not generate video.")

# ---------------- RUN Wav2Lip -----------------
print("💋 Running Wav2Lip for realistic lip sync...")
final_output = OUTPUT_DIR / "final_output.mp4"
cmd_wav2lip = [
    "python3", f"{WAV2LIP_DIR}/inference.py",
    "--checkpoint_path", f"{WAV2LIP_DIR}/checkpoints/wav2lip.pth",
    "--face", str(generated_video),
    "--audio", str(VOICE_FILE),
    "--outfile", str(final_output)
]
subprocess.run(cmd_wav2lip)

print(f"✅ Final talking head video ready: {final_output}")

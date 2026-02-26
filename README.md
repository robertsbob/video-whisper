# Media Whisper Transcriber

A Python tool that transcribes video and audio files using either OpenAI's Whisper API or a fully local GPU-accelerated Whisper model (faster-whisper). Supports large files by automatically splitting audio into chunks when needed.

## Features

- Transcribes both **video** and **audio** files
- **Two transcription modes:**
  - **OpenAI API** — cloud-based, requires API key
  - **Local GPU** — fully offline using faster-whisper with CUDA (large-v3 model, float16)
- Extracts audio from video formats using ffmpeg
- Automatically handles large files by splitting into chunks (API mode)
- Batch processes entire folders
- **Checkpoint/resume** — survives crashes, resumes where it left off
- Prompts before overwriting existing transcriptions (configurable: ask/skip/overwrite)
- Saves transcriptions as text files
- Detailed logging for troubleshooting

## Prerequisites

1. **Python 3.9+** with required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **ffmpeg** - Must be installed and available in your system PATH
   - Windows: Download from https://ffmpeg.org/
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

3. **For API mode:** OpenAI API Key from https://platform.openai.com/

4. **For local mode:** NVIDIA GPU with CUDA support and cuDNN installed

## Setup

1. Create a `.env` file in the project root with your settings:
   ```env
   # --- Transcription mode: "local" or "api" ---
   TRANSCRIPTION_MODE=local

   # --- OpenAI API settings (only needed when TRANSCRIPTION_MODE=api) ---
   OPENAI_API_KEY=your_key_here

   # --- Local Whisper settings (only needed when TRANSCRIPTION_MODE=local) ---
   WHISPER_MODEL=large-v3          # tiny | base | small | medium | large-v3 | large-v3-turbo
   WHISPER_DEVICE=cuda             # cuda | cpu
   WHISPER_COMPUTE_TYPE=float16    # float16 | int8 | float32
   WHISPER_BEAM_SIZE=5             # higher = more accurate but slower
   WHISPER_BATCH_SIZE=16            # >1 = batched inference (better GPU usage). 1 = sequential
   WHISPER_LANGUAGE=               # empty = auto-detect, or e.g. "en", "es", "fr"

   # --- Folders ---
   INPUT_FOLDER=in                 # folder containing your media files
   OUTPUT_FOLDER=video_transcriptions
   TEMP_AUDIO_FOLDER=temp_extracted_audio_from_videos

   # --- Audio extraction ---
   MP3_BITRATE_KBPS=96              # 96 is plenty for speech; lower = smaller temp files

   # --- Privacy ---
   OFFLINE_MODE=false              # true = block ALL network connections (local mode only)

   # --- Overwrite behaviour ---
   OVERWRITE=ask                   # ask | skip | overwrite

   # --- Logging ---
   LOG_LEVEL=INFO                  # DEBUG | INFO | WARNING | ERROR
   ```

2. Place your video/audio files in the input folder (default: `in/`).

3. **For offline use:** Pre-download the model first (requires internet):
   ```bash
   python transcribe.py --download-model
   ```
   > **Note:** `--download-model` always allows network access, even if `OFFLINE_MODE=true` is set.
   > This is the only command that bypasses offline mode — it needs internet to fetch the model.

   Models are cached by Hugging Face in:
   - **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`
   - **Linux/macOS:** `~/.cache/huggingface/hub/`
   - Override with the `HF_HOME` environment variable.

   Then set `OFFLINE_MODE=true` in `.env` for fully air-gapped transcription.

## Usage

Run the script:
```bash
python transcribe.py
```

All settings come from `.env` by default. Any setting can be overridden via CLI arguments:

```bash
# Transcribe a specific folder
python transcribe.py -i meetings/

# Transcribe a single file
python transcribe.py -i recording.mp4

# Override mode and model from CLI
python transcribe.py -i in/ --mode local --model large-v3

# Use API mode with key from CLI
python transcribe.py -i in/ --mode api --api-key sk-...

# Pre-download a model for offline use
python transcribe.py --download-model
python transcribe.py --download-model --model large-v3-turbo

# Full offline transcription
python transcribe.py -i in/ --offline

# Skip already-transcribed files without prompting
python transcribe.py -i in/ --skip

# Overwrite all existing transcriptions
python transcribe.py -i in/ --overwrite

# Ignore checkpoint and start fresh
python transcribe.py -i in/ --no-resume
```

### CLI Options

| Argument | Description |
|---|---|
| `input` (positional) | Path to a media file or folder |
| `-i`, `--input-path` | Same as positional input |
| `-m`, `--mode` | `local` or `api` |
| `-o`, `--output` | Output folder for transcriptions |
| `--model` | Whisper model name (e.g. `large-v3`, `medium`, `small`) |
| `--device` | `cuda` or `cpu` |
| `--compute-type` | `float16`, `int8`, or `float32` |
| `--beam-size` | Beam size (integer, higher = more accurate) |
| `--batch-size` | Batch size for GPU. >1 = batched mode (default: 16). 1 = sequential |
| `--language` | Language code (e.g. `en`, `es`). Empty = auto-detect |
| `--api-key` | OpenAI API key |
| `--offline` | Block all network connections |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `--download-model` | Download model for offline use, then exit (bypasses offline mode) |
| `--overwrite` | Overwrite existing transcriptions without prompting |
| `--skip` | Skip already-transcribed files (no prompt) |
| `--no-resume` | Ignore saved checkpoint, start from scratch |

If no input path is given (CLI or .env), the script falls back to `INPUT_FOLDER` from `.env`. If that doesn't exist, you'll be prompted.

## Supported Formats

**Video:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.mpeg`, `.mpg`, `.wmv`, `.ts`, `.m2ts`

**Audio:** `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

## File Structure

- `transcribe.py` - Main script
- `video_transcriptions/` - Output transcription files (ignored by git)
- `temp_extracted_audio_from_videos/` - Temporary audio files (auto-cleaned)

## Configuration

All settings are in `.env` (see `.env.example`). CLI arguments override `.env` values.
No code editing is needed.

## Privacy / Offline Mode

When `OFFLINE_MODE=true` in `.env`:
- All outgoing network connections are blocked at the socket level
- Only `TRANSCRIPTION_MODE=local` is allowed (API mode will refuse to start)
- The model must already be downloaded (use `python transcribe.py --download-model` first)
- Video, audio, and transcription data **never leaves your device**

> `--download-model` is the **only** operation that bypasses `OFFLINE_MODE`.
> Once the model is cached, you can run fully air-gapped.

## License

MIT License - see LICENSE file for details.
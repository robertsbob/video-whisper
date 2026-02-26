#!/usr/bin/env python3
"""
Media Whisper Transcriber
Transcribes video and audio files using OpenAI Whisper API or local GPU model (faster-whisper).
All settings are read from .env file — no code editing needed.
"""

import os
import re
import sys
import json
import socket
import argparse
import subprocess
import shutil
import logging
import importlib.util

# ---------------------------------------------------------------------------
# Auto-detect NVIDIA CUDA DLLs installed via pip (Windows)
# CTranslate2 needs cublas/cudnn but pip puts them in site-packages/nvidia/.
# We preload the DLLs into the process via ctypes so CTranslate2 finds them.
# This does NOT modify PATH or any environment variables.
# ---------------------------------------------------------------------------
def _add_nvidia_dll_paths():
    if sys.platform != "win32":
        return
    import ctypes
    try:
        spec = importlib.util.find_spec("nvidia")
        if spec is None or spec.submodule_search_locations is None:
            return
        # DLLs we need to preload for CTranslate2 (order matters — dependencies first)
        needed = [
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
        ]
        dll_map = {}  # filename -> full path
        for nvidia_root in spec.submodule_search_locations:
            for dirpath, _dirs, filenames in os.walk(nvidia_root):
                for fn in filenames:
                    if fn.lower() in [n.lower() for n in needed]:
                        dll_map[fn.lower()] = os.path.join(dirpath, fn)
        for dll_name in needed:
            full_path = dll_map.get(dll_name.lower())
            if full_path and os.path.isfile(full_path):
                try:
                    ctypes.WinDLL(full_path)
                except OSError:
                    pass
    except Exception:
        pass  # non-critical — user may have system-level CUDA installed

_add_nvidia_dll_paths()

from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def _setup_logging(level_name: str = "INFO") -> logging.Logger:
    """Configure root logger and return the app logger."""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    return logging.getLogger("transcriber")

# ---------------------------------------------------------------------------
# .env-driven configuration (defaults shown)
# ---------------------------------------------------------------------------
load_dotenv()

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_bool(key: str, default: bool = False) -> bool:
    val = _env(key, str(default)).lower()
    return val in ("true", "1", "yes")

def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default

# General
TRANSCRIPTION_MODE  = _env("TRANSCRIPTION_MODE", "local")       # "local" or "api"
INPUT_FOLDER        = _env("INPUT_FOLDER", "")  # no default — must be set via .env or CLI
OUTPUT_FOLDER       = _env("OUTPUT_FOLDER", "video_transcriptions")
TEMP_AUDIO_FOLDER   = _env("TEMP_AUDIO_FOLDER", "temp_extracted_audio_from_videos")
LOG_LEVEL           = _env("LOG_LEVEL", "INFO")
MP3_BITRATE_KBPS    = _env_int("MP3_BITRATE_KBPS", 96)

# OpenAI API settings
OPENAI_API_KEY      = _env("OPENAI_API_KEY")
WHISPER_API_FILE_SIZE_LIMIT = 25 * 1024 * 1024   # 25 MB (API hard limit)
SAFE_CHUNK_SIZE_BYTES       = 24 * 1024 * 1024   # 24 MB

# Local Whisper (faster-whisper) settings
WHISPER_MODEL        = _env("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE       = _env("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_BEAM_SIZE    = _env_int("WHISPER_BEAM_SIZE", 5)
WHISPER_BATCH_SIZE   = _env_int("WHISPER_BATCH_SIZE", 16)
WHISPER_LANGUAGE     = _env("WHISPER_LANGUAGE", "")  # empty = auto-detect

# Privacy
OFFLINE_MODE = _env_bool("OFFLINE_MODE", False)

# Overwrite behaviour: "ask", "skip", "overwrite"
OVERWRITE = _env("OVERWRITE", "ask")

# Supported extensions
SUPPORTED_VIDEO_EXTENSIONS = (
    '.mp4', '.mkv', '.avi', '.mov', '.webm',
    '.flv', '.mpeg', '.mpg', '.wmv', '.ts', '.m2ts'
)
SUPPORTED_AUDIO_EXTENSIONS = (
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'
)

logger = _setup_logging(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Offline mode — block all network access for this process
# ---------------------------------------------------------------------------
_original_socket_connect = socket.socket.connect

def _blocked_connect(self, *args, **kwargs):
    raise OSError("Network access is blocked (OFFLINE_MODE=true). "
                  "All transcription is local-only.")

def enforce_offline_mode():
    """Monkey-patch socket.connect so no data can leave the device."""
    socket.socket.connect = _blocked_connect
    logger.info("OFFLINE_MODE enabled — all network connections blocked for this process.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    if name is None:
        name = "untitled"
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_ ')
    return name or "untitled_file"

def is_audio_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)

def is_video_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS)

def is_media_file(path: str) -> bool:
    return is_audio_file(path) or is_video_file(path)

# ---------------------------------------------------------------------------
# Checkpoint system — resume after crash
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = os.path.join(TEMP_AUDIO_FOLDER, ".checkpoint.json")

def _load_checkpoint() -> dict:
    """Load checkpoint data. Returns dict with 'completed' list and 'input_path'."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Corrupt checkpoint file — starting fresh.")
    return {"completed": [], "input_path": None}

def _save_checkpoint(data: dict):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE) or ".", exist_ok=True)
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        logger.warning("Could not save checkpoint.")

def _clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logger.debug("Checkpoint file removed.")
        except OSError:
            pass

def _should_process(base_name: str, overwrite: str) -> bool:
    """Check if a file should be processed based on existing output and overwrite setting."""
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
    if not os.path.exists(output_path):
        return True

    if overwrite == "overwrite":
        logger.info("Overwriting existing transcription: %s", output_path)
        return True
    elif overwrite == "skip":
        logger.info("Skipping (already transcribed): %s", output_path)
        return False
    else:  # "ask"
        answer = input(f"  '{base_name}.txt' already exists. Overwrite? [y/N]: ").strip().lower()
        return answer in ("y", "yes")

# ---------------------------------------------------------------------------
# Audio extraction (video → mp3) using ffmpeg
# ---------------------------------------------------------------------------

def extract_audio_from_video(video_file_path: str) -> str | None:
    """Extract audio from a video file to MP3.  Returns path or None on failure."""
    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
    base = sanitize_filename(video_file_path)
    output_path = os.path.join(TEMP_AUDIO_FOLDER, f"{base}.mp3")

    if os.path.exists(output_path):
        logger.info("Reusing existing extracted audio: %s", output_path)
        return output_path

    cmd = [
        'ffmpeg', '-i', video_file_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-ab', f'{MP3_BITRATE_KBPS}k',
        '-ar', '44100',
        '-y', output_path
    ]
    logger.info("Extracting audio: %s → %s", video_file_path, output_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info("Audio extraction successful.")
            return output_path
        logger.error("ffmpeg failed for %s\nstdout: %s\nstderr: %s",
                      video_file_path, result.stdout, result.stderr)
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install it and add to PATH.")
        return None
    except Exception:
        logger.exception("Unexpected error during audio extraction")
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        return None

def prepare_audio_path(media_path: str) -> str | None:
    """
    For video files, extract audio to temp MP3.
    For audio files, return the path directly (no extraction needed).
    """
    if is_audio_file(media_path):
        logger.info("File is already audio — using directly: %s", media_path)
        return media_path
    elif is_video_file(media_path):
        return extract_audio_from_video(media_path)
    return None

# ---------------------------------------------------------------------------
# OpenAI API transcription
# ---------------------------------------------------------------------------

def _transcribe_single_api(audio_path: str, api_key: str) -> str | None:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    try:
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info("Sending %s (%.2f MB) to Whisper API …", os.path.basename(audio_path), size_mb)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )
        return transcript
    except Exception:
        logger.exception("Whisper API error for %s", audio_path)
        return None

def _split_and_transcribe_api(audio_path: str, api_key: str) -> str | None:
    base = sanitize_filename(audio_path)
    chunks_dir = os.path.join(TEMP_AUDIO_FOLDER, f"{base}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    logger.info("Splitting large audio: %s", audio_path)
    try:
        audio = AudioSegment.from_mp3(audio_path)
    except Exception:
        logger.exception("pydub could not load %s", audio_path)
        if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir)
        return None

    bps = (MP3_BITRATE_KBPS * 1000) / 8.0
    if bps <= 0:
        logger.error("Invalid MP3_BITRATE_KBPS: %d", MP3_BITRATE_KBPS)
        if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir)
        return None

    chunk_ms = int((SAFE_CHUNK_SIZE_BYTES / bps) * 0.95 * 1000)
    if chunk_ms <= 1000:
        logger.error("Calculated chunk length too small (%d ms)", chunk_ms)
        if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir)
        return None

    logger.info("Chunk target: %.1f s", chunk_ms / 1000)
    chunks = make_chunks(audio, chunk_ms)
    parts = []

    for i, seg in enumerate(chunks):
        chunk_path = os.path.join(chunks_dir, f"chunk_{base}_{i}.mp3")
        logger.info("Exporting chunk %d/%d: %s", i + 1, len(chunks), chunk_path)
        try:
            seg.export(chunk_path, format="mp3", bitrate=f"{MP3_BITRATE_KBPS}k")
        except Exception:
            logger.exception("Failed to export chunk %s", chunk_path)
            break

        if os.path.getsize(chunk_path) >= WHISPER_API_FILE_SIZE_LIMIT:
            logger.error("Chunk %s too large — aborting", chunk_path)
            break

        text = _transcribe_single_api(chunk_path, api_key)
        try: os.remove(chunk_path)
        except OSError: pass

        if text is None:
            logger.error("Chunk %d/%d failed — aborting file", i + 1, len(chunks))
            break
        parts.append(text)
    else:
        # loop completed without break
        if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir, ignore_errors=True)
        return " ".join(parts) if parts else None

    if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir, ignore_errors=True)
    return None

def transcribe_api(audio_path: str, api_key: str) -> str | None:
    if not api_key:
        logger.error("No OpenAI API key provided.")
        return None
    if not audio_path or not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return None

    size = os.path.getsize(audio_path)
    if size < WHISPER_API_FILE_SIZE_LIMIT * 0.98:
        return _transcribe_single_api(audio_path, api_key)
    else:
        return _split_and_transcribe_api(audio_path, api_key)

# ---------------------------------------------------------------------------
# Local (faster-whisper) transcription — fully offline, GPU-accelerated
# ---------------------------------------------------------------------------

_local_model = None  # cached across files
_batched_model = None  # cached BatchedInferencePipeline

def _get_local_model():
    """Load faster-whisper model (cached for the session)."""
    global _local_model, _batched_model
    if _local_model is not None:
        return _local_model, _batched_model

    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        logger.error("faster-whisper is not installed.  Run:  pip install faster-whisper")
        sys.exit(1)

    logger.info("Loading local Whisper model: %s  device=%s  compute=%s",
                WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)

    try:
        _local_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if OFFLINE_MODE or "network" in err_msg or "connection" in err_msg or "localentrynotfounderror" in type(e).__name__.lower():
            logger.error("Model '%s' is not available locally. "
                         "Download it first:  python transcribe.py --download-model",
                         WHISPER_MODEL)
        else:
            logger.error("Failed to load model '%s': %s", WHISPER_MODEL, e)
            logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

    if WHISPER_BATCH_SIZE > 1:
        _batched_model = BatchedInferencePipeline(model=_local_model)
        logger.info("Model loaded successfully (batched mode, batch_size=%d).", WHISPER_BATCH_SIZE)
    else:
        logger.info("Model loaded successfully (sequential mode).")

    return _local_model, _batched_model

def transcribe_local(audio_path: str) -> str | None:
    """Transcribe an audio file using the local faster-whisper model on GPU."""
    if not audio_path or not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return None

    from tqdm import tqdm

    model, batched = _get_local_model()
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info("Transcribing locally: %s (%.2f MB)", os.path.basename(audio_path), size_mb)

    try:
        kwargs: dict = {"beam_size": WHISPER_BEAM_SIZE}
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE

        # Use batched pipeline if available (much better GPU utilization)
        if batched and WHISPER_BATCH_SIZE > 1:
            kwargs["batch_size"] = WHISPER_BATCH_SIZE
            segments, info = batched.transcribe(audio_path, **kwargs)
        else:
            segments, info = model.transcribe(audio_path, **kwargs)

        total_duration = info.duration
        logger.info("Detected language: %s (probability %.2f) — duration: %.0fs",
                     info.language, info.language_probability, total_duration)

        parts = []
        # Progress bar based on audio seconds processed
        with tqdm(total=int(total_duration), unit="s", desc="Transcribing",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]") as pbar:
            last_end = 0
            for seg in segments:
                logger.debug("[%.2fs → %.2fs] %s", seg.start, seg.end, seg.text.strip())
                parts.append(seg.text.strip())
                # Update progress to segment end position
                advance = int(seg.end) - last_end
                if advance > 0:
                    pbar.update(advance)
                    last_end = int(seg.end)
            # Fill remaining
            remaining = int(total_duration) - last_end
            if remaining > 0:
                pbar.update(remaining)

        text = " ".join(parts)
        logger.info("Transcription complete — %d characters", len(text))
        return text
    except Exception:
        logger.exception("Local transcription failed for %s", audio_path)
        return None

# ---------------------------------------------------------------------------
# Download model (for offline setup)
# ---------------------------------------------------------------------------

def download_model():
    """Pre-download the configured model so it's available in offline mode."""
    logger.info("Downloading model '%s' for offline use …", WHISPER_MODEL)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper is not installed.  Run:  pip install faster-whisper")
        sys.exit(1)

    try:
        # Loading the model triggers the download
        WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        logger.info("Model '%s' downloaded and cached successfully.", WHISPER_MODEL)
    except Exception:
        logger.exception("Failed to download model '%s'", WHISPER_MODEL)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Save transcription
# ---------------------------------------------------------------------------

def save_transcription(text: str, base_name: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved transcription: %s", path)
    except Exception:
        logger.exception("Failed to save transcription to %s", path)

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_single_file(file_path: str, mode: str, api_key: str = "",
                        overwrite: str = "ask"):
    """Process a single media file."""
    if not os.path.isfile(file_path):
        logger.error("File not found: %s", file_path)
        return
    if not is_media_file(file_path):
        logger.error("Unsupported file type: %s", file_path)
        logger.info("Supported: %s %s",
                     ", ".join(SUPPORTED_VIDEO_EXTENSIONS),
                     ", ".join(SUPPORTED_AUDIO_EXTENSIONS))
        return

    name = os.path.basename(file_path)
    base = sanitize_filename(name)

    if not _should_process(base, overwrite):
        return

    logger.info("--- Processing: %s ---", name)

    audio_path = prepare_audio_path(file_path)
    is_temp_audio = audio_path != file_path

    if not audio_path or not os.path.exists(audio_path):
        logger.error("Could not obtain audio for '%s'.", name)
        return

    if mode == "local":
        text = transcribe_local(audio_path)
    else:
        text = transcribe_api(audio_path, api_key)

    if text is not None:
        save_transcription(text, base)
    else:
        logger.error("Transcription failed for '%s'.", name)

    if is_temp_audio:
        try:
            os.remove(audio_path)
            logger.debug("Removed temp audio: %s", audio_path)
        except OSError as e:
            logger.warning("Could not remove temp audio %s: %s", audio_path, e)


def process_folder(folder_path: str, mode: str, api_key: str = "",
                   overwrite: str = "ask", resume: bool = True):
    """Process all media files in a folder with checkpoint/resume support."""
    if not os.path.isdir(folder_path):
        logger.error("Folder not found: %s", folder_path)
        return

    media_files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and is_media_file(f)
    )

    if not media_files:
        logger.warning("No supported media files in '%s'.", folder_path)
        logger.info("Supported: %s %s",
                     ", ".join(SUPPORTED_VIDEO_EXTENSIONS),
                     ", ".join(SUPPORTED_AUDIO_EXTENSIONS))
        return

    # --- Checkpoint: detect previous incomplete run ---
    checkpoint = _load_checkpoint()
    already_done = set(checkpoint.get("completed", []))
    prev_input = checkpoint.get("input_path")

    if already_done and prev_input == os.path.abspath(folder_path) and resume:
        remaining = [f for f in media_files if os.path.basename(f) not in already_done]
        if remaining and len(remaining) < len(media_files):
            logger.info("Found checkpoint: %d/%d files already done.",
                         len(already_done), len(media_files))
            if overwrite != "overwrite":
                answer = input("  Resume from checkpoint? [Y/n]: ").strip().lower()
                if answer in ("n", "no"):
                    already_done = set()
                    logger.info("Starting from scratch.")
            else:
                already_done = set()
        elif not remaining:
            logger.info("All files already processed (checkpoint). Nothing to do.")
            _clear_checkpoint()
            return
    else:
        already_done = set()

    # Init fresh checkpoint for this run
    checkpoint_data = {
        "input_path": os.path.abspath(folder_path),
        "completed": list(already_done),
    }
    _save_checkpoint(checkpoint_data)

    logger.info("Found %d media file(s) in '%s'.", len(media_files), folder_path)

    for i, media_path in enumerate(media_files, 1):
        name = os.path.basename(media_path)
        base = sanitize_filename(name)

        # Skip if completed in a previous run (resume)
        if name in already_done:
            logger.info("[%d/%d] Skipping (checkpoint): %s", i, len(media_files), name)
            continue

        # Skip if output already exists
        if not _should_process(base, overwrite):
            checkpoint_data["completed"].append(name)
            _save_checkpoint(checkpoint_data)
            continue

        logger.info("\n--- [%d/%d] %s ---", i, len(media_files), name)

        audio_path = prepare_audio_path(media_path)
        is_temp_audio = audio_path != media_path

        if not audio_path or not os.path.exists(audio_path):
            logger.error("Could not obtain audio for '%s'. Skipping.", name)
            continue

        if mode == "local":
            text = transcribe_local(audio_path)
        else:
            text = transcribe_api(audio_path, api_key)

        if text is not None:
            save_transcription(text, base)
        else:
            logger.error("Transcription failed for '%s'.", name)

        if is_temp_audio:
            try:
                os.remove(audio_path)
                logger.debug("Removed temp audio: %s", audio_path)
            except OSError as e:
                logger.warning("Could not remove temp audio %s: %s", audio_path, e)

        # Mark done in checkpoint
        checkpoint_data["completed"].append(name)
        _save_checkpoint(checkpoint_data)

    # All done — remove checkpoint
    _clear_checkpoint()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Media Whisper Transcriber — transcribe video/audio files via OpenAI API or local GPU model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python transcribe.py                              # use .env defaults
  python transcribe.py -i meetings/                 # transcribe a folder
  python transcribe.py -i recording.mp4             # transcribe a single file
  python transcribe.py -i in/ --mode local --model large-v3
  python transcribe.py -i in/ --mode api --api-key sk-...
  python transcribe.py --download-model             # pre-download model for offline use
  python transcribe.py --download-model --model large-v3-turbo

All CLI arguments override .env settings. If no input is given, the INPUT_FOLDER
from .env is used (default: "in"). If that folder doesn't exist, you'll be prompted.
"""
    )

    parser.add_argument("input", nargs="?", default=None,
                        help="Path to a media file or folder. Overrides INPUT_FOLDER from .env.")
    parser.add_argument("-i", "--input-path", default=None,
                        help="Same as positional input — path to a media file or folder.")
    parser.add_argument("-m", "--mode", choices=["local", "api"], default=None,
                        help="Transcription mode (default: from .env or 'local').")
    parser.add_argument("-o", "--output", default=None,
                        help="Output folder for transcriptions (default: from .env or 'video_transcriptions').")
    parser.add_argument("--model", default=None,
                        help="Whisper model name, e.g. large-v3, medium, small (default: from .env).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device for local model (default: from .env or 'cuda').")
    parser.add_argument("--compute-type", default=None,
                        help="Compute type: float16, int8, float32 (default: from .env or 'float16').")
    parser.add_argument("--beam-size", type=int, default=None,
                        help="Beam size for local model (default: from .env or 5).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for local model. >1 enables batched inference for "
                             "better GPU utilization (default: from .env or 16). Set to 1 for sequential.")
    parser.add_argument("--language", default=None,
                        help="Language code, e.g. en, es, fr. Empty = auto-detect (default: from .env).")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (default: from .env).")
    parser.add_argument("--offline", action="store_true", default=None,
                        help="Block all network connections (local mode only).")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
                        help="Log level (default: from .env or 'INFO').")
    parser.add_argument("--download-model", action="store_true",
                        help="Download the configured Whisper model for offline use, then exit. "
                             "Bypasses OFFLINE_MODE (needs internet). "
                             "Models are cached in ~/.cache/huggingface/hub/ (override with HF_HOME).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing transcriptions without prompting.")
    parser.add_argument("--skip", action="store_true",
                        help="Skip files that already have a transcription (no prompt).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any saved checkpoint and start from scratch.")

    args = parser.parse_args()

    # --- Apply CLI overrides to globals ---
    global TRANSCRIPTION_MODE, OUTPUT_FOLDER, WHISPER_MODEL, WHISPER_DEVICE
    global WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE, WHISPER_BATCH_SIZE, WHISPER_LANGUAGE
    global OPENAI_API_KEY, OFFLINE_MODE, LOG_LEVEL, OVERWRITE, logger

    if args.log_level:
        LOG_LEVEL = args.log_level
        logger = _setup_logging(LOG_LEVEL)
    if args.mode:
        TRANSCRIPTION_MODE = args.mode
    if args.output:
        OUTPUT_FOLDER = args.output
    if args.model:
        WHISPER_MODEL = args.model
    if args.device:
        WHISPER_DEVICE = args.device
    if args.compute_type:
        WHISPER_COMPUTE_TYPE = args.compute_type
    if args.beam_size is not None:
        WHISPER_BEAM_SIZE = args.beam_size
    if args.batch_size is not None:
        WHISPER_BATCH_SIZE = args.batch_size
    if args.language is not None:
        WHISPER_LANGUAGE = args.language
    if args.api_key:
        OPENAI_API_KEY = args.api_key
    if args.offline:
        OFFLINE_MODE = True
    if args.overwrite:
        OVERWRITE = "overwrite"
    elif args.skip:
        OVERWRITE = "skip"

    resume = not args.no_resume

    # Resolve input path: CLI > .env > prompt
    input_path = args.input or args.input_path  # positional takes priority over -i

    # Download-only mode
    if args.download_model:
        download_model()
        sys.exit(0)

    # Enforce offline mode before any network-capable code runs
    if OFFLINE_MODE:
        enforce_offline_mode()
        if TRANSCRIPTION_MODE == "api":
            logger.error("OFFLINE_MODE is incompatible with TRANSCRIPTION_MODE=api. "
                          "Use --mode local or disable OFFLINE_MODE.")
            sys.exit(1)

    print("Media Whisper Transcriber")
    print("=" * 60)

    mode = TRANSCRIPTION_MODE.lower()
    api_key = OPENAI_API_KEY

    if mode == "api":
        if not api_key:
            logger.warning("OPENAI_API_KEY not set.")
            api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
            if not api_key:
                logger.error("No API key — cannot use API mode.")
                sys.exit(1)
        logger.info("Mode: OpenAI API")
    elif mode == "local":
        logger.info("Mode: Local (faster-whisper)  model=%s  device=%s  compute=%s  beam=%d  batch=%d",
                     WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE, WHISPER_BATCH_SIZE)
        if OFFLINE_MODE:
            logger.info("Privacy: OFFLINE_MODE active — no network connections allowed.")
    else:
        logger.error("Unknown mode '%s'. Use 'local' or 'api'.", mode)
        sys.exit(1)

    # Resolve input: if not given via CLI, fall back to .env INPUT_FOLDER, then prompt
    if not input_path:
        input_path = INPUT_FOLDER
    if not input_path or not os.path.exists(input_path):
        input_path = input("\nEnter path to a media file or folder: ").strip()

    if not os.path.exists(input_path):
        logger.error("Path not found: %s", input_path)
        sys.exit(1)

    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Single file or folder
    if os.path.isfile(input_path):
        process_single_file(input_path, mode, api_key, overwrite=OVERWRITE)
    elif os.path.isdir(input_path):
        process_folder(input_path, mode, api_key, overwrite=OVERWRITE, resume=resume)
    else:
        logger.error("Path is neither a file nor a folder: %s", input_path)
        sys.exit(1)

    # Cleanup empty temp folder
    try:
        if os.path.exists(TEMP_AUDIO_FOLDER) and not os.listdir(TEMP_AUDIO_FOLDER):
            os.rmdir(TEMP_AUDIO_FOLDER)
            logger.debug("Removed empty temp folder: %s", TEMP_AUDIO_FOLDER)
    except OSError:
        pass

    print("\n--- All processing complete. ---")

if __name__ == "__main__":
    main()

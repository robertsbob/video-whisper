# Video Whisper Transcriber

A Python tool that extracts audio from video files and transcribes them using OpenAI's Whisper API. Supports large video files by automatically splitting audio into chunks when needed.

## Features

- Extracts audio from various video formats using ffmpeg
- Automatically handles large files by splitting into chunks
- Uses OpenAI Whisper API for accurate transcription
- Batch processes entire folders of videos
- Saves transcriptions as text files

## Prerequisites

1. **Python 3.x** with required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **ffmpeg** - Must be installed and available in your system PATH
   - Windows: Download from https://ffmpeg.org/
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

3. **OpenAI API Key** - Get one from https://platform.openai.com/

## Setup

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY="your_actual_api_key_here"
   ```

2. Place your video files in a folder (the `in/` folder contains example videos)

## Usage

Run the script:
```bash
python transcribe.py
```

The script will:
1. Prompt for your OpenAI API key (if not in .env)
2. Ask for the path to your video folder
3. Process all supported video files in that folder
4. Save transcriptions to the `video_transcriptions/` folder

## Supported Video Formats

- `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`
- `.flv`, `.mpeg`, `.mpg`, `.wmv`, `.ts`, `.m2ts`

## File Structure

- `transcribe.py` - Main script
- `in/` - Input video files (ignored by git)
- `video_transcriptions/` - Output transcription files (ignored by git)
- `temp_extracted_audio_from_videos/` - Temporary audio files (auto-cleaned)

## Configuration

The script automatically handles files larger than 25MB by splitting them into chunks. You can modify these settings at the top of `transcribe.py`:

- `WHISPER_API_FILE_SIZE_LIMIT` - OpenAI API file size limit
- `SAFE_CHUNK_SIZE_BYTES` - Target chunk size for splitting
- `MP3_BITRATE_KBPS` - Audio bitrate for extraction

## License

MIT License - see LICENSE file for details.
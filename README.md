# Whisper_transcript_and_rename

A simple script that uses OpenAI's Whisper model to transcribe a folder containing multiple voice lines and renames the files based on their corresponding transcriptions. If a transcription exceeds the maximum file length, the script truncates the filename to fit within the allowed limit.

---

## Features
- **Automated transcription**: Transcribes multiple audio files in a folder using Whisper.
- **Automatic renaming**: Renames files based on transcriptions.
- **Filename truncation**: Automatically truncates transcriptions if they exceed the max file name length.

---

## Requirements

1. **Python 3.x**
2. **Torch** (with GPU support recommended for better performance)
3. **Whisper** (OpenAI's pre-trained model)
4. **ffmpeg** (for audio file processing)

---

## Installation

### 1. Install Python and Torch (with GPU support)
Follow the instructions from the [official PyTorch website](https://pytorch.org/get-started/locally/) to install Torch with GPU support. Example for CUDA installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 2. Install Whisper
Use the following command to install Whisper:

```bash
pip install git+https://github.com/openai/whisper.git
```

### 3. Install `ffmpeg`
Whisper requires `ffmpeg` to process audio files.

#### On Windows:
- Download `ffmpeg` from [ffmpeg.org](https://ffmpeg.org/download.html).
- Extract the folder and add the `bin` folder to your system's PATH. 

#### On macOS (with Homebrew):
```bash
brew install ffmpeg
```

#### On Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/dahshury/Whisper_transcript_and_rename.git
cd Whisper_transcript_and_rename
```

### 2. Transcribe and rename files

Run the script using the command below, replacing `<input_folder>` with the path to your folder of audio files:

```bash
python rename_transcripts.py --input_folder <input_folder> --max_length 255
```

- `--input_folder`: Path to the folder containing the audio files.
- `--max_length`: (Optional) The maximum allowed filename length. Default is set to 255.

### 3. Example

```bash
python rename_transcripts.py --input_folder ./voice_lines --max_length 100
```

This will transcribe all audio files in the `voice_lines` folder and rename them based on the transcriptions, with filenames truncated to 100 characters if necessary.

---

## Supported File Formats

The script supports various audio file formats compatible with `ffmpeg`, including:
- `.wav`
- `.mp3`
- `.flac`
- `.m4a`

---

## Notes

- The default maximum file name length is 255 characters (suitable for most filesystems).
- Transcriptions may vary in accuracy based on audio quality and language used.

---

## Contributing

Feel free to submit issues or pull requests to contribute to this project.

---

## License

This project is licensed under the MIT License.
```

import os
import argparse
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import torch

# Load the Whisper model with appropriate settings
model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

# Function to transcribe and rename files
def transcribe_and_rename_files(folder_path, max_length=255):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {file_path}")

        if filename.lower().endswith(('.wav', '.mp3')):
            try:
                # Perform transcription
                outputs = model(
                    file_path,
                    chunk_length_s=30,
                    batch_size=24,
                    return_timestamps=True,
                )
                transcription = outputs['text']
                valid_filename = ''.join(c for c in transcription if c.isalnum() or c in (' ', '_')).rstrip()
                file_extension = os.path.splitext(filename)[1]
                new_file_path = os.path.join(folder_path, valid_filename + file_extension)
                
                # Truncate the file name if necessary
                if len(new_file_path) > max_length:
                    valid_filename = valid_filename[:max_length - len(folder_path) - len(file_extension) - 1]
                    new_file_path = os.path.join(folder_path, valid_filename + file_extension)
                
                os.rename(file_path, new_file_path)
                print(f"Renamed {filename} to {valid_filename + file_extension}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Main block to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and rename audio files based on transcriptions")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing audio files")
    parser.add_argument('--max_length', type=int, default=255, help="Maximum length for renamed files (default: 255)")
    
    args = parser.parse_args()
    
    transcribe_and_rename_files(args.input_folder, args.max_length)

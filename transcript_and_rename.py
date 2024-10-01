# Install torch gpu and add download ffmpeg and add it to path 
import os
import subprocess
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import torch

model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="cuda:0",
                model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            )

def transcribe_and_rename_files(folder_path):
    # Maximum filename length for most filesystems
    MAX_PATH_LENGTH = 255

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(file_path)

        # Check if the file is a .wav or .mp3 file
        if filename.lower().endswith(('.wav', '.mp3')):
            try:
                outputs = model(
                                file_path,
                                chunk_length_s=30,
                                batch_size=24,
                                return_timestamps=True,
                            )

                # Get the transcription from the command output
                transcription = outputs['text']

                # Create a valid filename from the transcription
                valid_filename = ''.join(c for c in transcription if c.isalnum() or c in (' ', '_')).rstrip()
                
                # Get the file extension
                file_extension = os.path.splitext(filename)[1]
                
                new_file_path = os.path.join(folder_path, valid_filename + file_extension)
                
                if len(new_file_path) > MAX_PATH_LENGTH:
                    valid_filename = valid_filename[:MAX_PATH_LENGTH-len(folder_path)-3]
                    new_file_path = os.path.join(folder_path, valid_filename + file_extension)
                    


                # Rename the file
                os.rename(file_path, new_file_path)

                print(f"Renamed {filename} to {valid_filename + file_extension}")
            except Exception as e:
                print(f"Unexpected error processing {filename}: {e}")

# Example usage
folder_path = "C:/Users/MASTER/Desktop/eva_project/sofia eva/new"
transcribe_and_rename_files(folder_path)

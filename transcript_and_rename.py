import os
import subprocess
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
def transcribe_and_rename_files(folder_path, max_length=250):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Process each file in the folder
    for dirpath, foldernames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
            else:
                continue
            # Check if the file is a .wav or .mp3
            if filename.lower().endswith(('.wav', '.mp3')):
                try:
                    # Perform transcription
                    outputs = model(
                        file_path,
                        chunk_length_s=30,
                        batch_size=24,
                        return_timestamps=True,
                    )

                    # Extract the transcription text
                    transcription = outputs['text']

                    # Create a valid filename from the transcription
                    valid_filename = ''.join(c for c in transcription if c.isalnum() or c in (' ', '_')).rstrip()
                    
                    # Get file extension
                    file_extension = os.path.splitext(filename)[1]
                    
                    # Create the full new file path
                    new_file_path = os.path.join(dirpath, valid_filename + file_extension)
                    
                    # Truncate the file name if it exceeds the max_length
                    if len(new_file_path) > max_length:
                        valid_filename = valid_filename[:max_length-len(folder_path)-len(file_extension)-1]
                        new_file_path = os.path.join(dirpath, valid_filename + file_extension)
                    
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {filename} to {valid_filename + file_extension}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Example usage
folder_path = "C:/Users/MASTER/Desktop/eva_project/gla_eva/original"
transcribe_and_rename_files(folder_path, max_length=100)

import os
import requests
from tqdm import tqdm
import onnxruntime as ort
import numpy as np
from transformers import AutoProcessor
import json
import librosa


class WhisperONNXTranscriber:
    def __init__(self, onnx_path, q=None if 'CUDAExecutionProvider' in ort.get_available_providers() else "q4f16"):
        self.onnx_path = onnx_path
        self.q = q

        # Subfolder for ONNX files
        self.onnx_folder = os.path.join(self.onnx_path, "onnx")

        self.encoder_name = "encoder_model.onnx" if self.q is None else f"encoder_model_{self.q}.onnx"
        self.decoder_name = "decoder_model.onnx" if self.q is None else f"decoder_model_{self.q}.onnx"

        # Ensure models are downloaded
        self.download_and_prepare_models()

        # Load ONNX models
        available_providers = ort.get_available_providers()

        self.encoder_session = ort.InferenceSession(
            os.path.join(self.onnx_folder, self.encoder_name),
            providers=available_providers
        )
        self.decoder_session = ort.InferenceSession(
            os.path.join(self.onnx_folder, self.decoder_name),
            providers=available_providers
        )

        # Load processor and configs
        self.processor = AutoProcessor.from_pretrained(self.onnx_path)

        # Load configuration files
        with open(os.path.join(self.onnx_path, "config.json"), 'r') as f:
            self.config = json.load(f)
        with open(os.path.join(self.onnx_path, "generation_config.json"), 'r') as f:
            self.generation_config = json.load(f)
        with open(os.path.join(self.onnx_path, "preprocessor_config.json"), 'r') as f:
            self.preprocessor_config = json.load(f)

    def download_and_prepare_models(self):
        """Downloads the model files if they don't exist."""
        os.makedirs(self.onnx_folder, exist_ok=True)

        repo_url = "https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/onnx/"
        files = [
            self.encoder_name,
            "encoder_model.onnx_data" if self.q is None else None,
            self.decoder_name
        ]

        config_files = [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "merges.txt",
            "vocab.json"
        ]

        # Download ONNX files
        for file_name in files:
            if file_name is not None:
                file_path = os.path.join(self.onnx_folder, file_name)
                if not os.path.exists(file_path):
                    print(f"File '{file_name}' not found. Downloading...")
                    file_url = repo_url + file_name
                    self.download_file_with_progress(file_url, file_path)
                else:
                    print(f"File '{file_name}' already exists. Skipping download.")

        # Download configuration files
        for config_file in config_files:
            config_path = os.path.join(self.onnx_path, config_file)
            config_url = f"https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/{config_file}"
            if not os.path.exists(config_path):
                print(f"Configuration file '{config_file}' not found. Downloading...")
                self.download_file_with_progress(config_url, config_path)
            else:
                print(f"Configuration file '{config_file}' already exists. Skipping download.")

    @staticmethod
    def download_file_with_progress(url, save_path):
        """Download a file from a URL with a progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(save_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

    def preprocess_audio(self, audio_path):
        # Get parameters from preprocessor_config
        sampling_rate = self.preprocessor_config.get('sampling_rate', 16000)
        chunk_length_s = self.preprocessor_config.get('chunk_length', 30)
        padding_value = self.preprocessor_config.get('padding_value', 0.0)
        return_attention_mask = self.preprocessor_config.get('return_attention_mask', False)

        # Load audio file using librosa
        audio, sr = librosa.load(audio_path, sr=sampling_rate)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        # Calculate number of samples for chunk_length_s
        chunk_samples = int(chunk_length_s * sampling_rate)

        # Pad or truncate audio to chunk_length_s
        if len(audio) > chunk_samples:
            audio = audio[:chunk_samples]
        else:
            # Pad with zeros or specified padding_value if audio is shorter than chunk_length_s
            padding = chunk_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant', constant_values=(padding_value, padding_value))
        # Process through whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="np",
            return_attention_mask=return_attention_mask
        )

        # Convert input features to float32
        input_features = inputs.input_features.astype(np.float32)

        return input_features

    def encode(self, input_features):
        # Run encoder
        encoder_outputs = self.encoder_session.run(
            None,
            {"input_features": input_features}
        )
        return encoder_outputs[0]

    def decode(self, encoder_hidden_states, attention_mask=None):
        # Initialize decoder inputs
        batch_size = encoder_hidden_states.shape[0]
        decoder_input_ids = np.array([[self.generation_config["decoder_start_token_id"]]] * batch_size, dtype=np.int64)

        output_ids = []
        max_length = self.generation_config.get("max_length", 448)

        for _ in range(max_length):
            # Prepare decoder inputs
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states.astype(np.float32)  # Ensure float32
            }

            # Run decoder
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            next_token_logits = decoder_outputs[0][:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1)

            # Append tokens
            output_ids.append(next_tokens)

            # Update decoder input ids
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, next_tokens[:, None]], axis=-1
            )

            # Check for end of sequence
            if next_tokens[0] == self.generation_config["eos_token_id"]:
                break

        return np.array(output_ids, dtype=np.int64).T  # Ensure int64

    def postprocess(self, output_ids):
        # Convert output_ids to list for the processor
        output_ids_list = output_ids.tolist()

        # Decode the predicted tokens to text
        transcription = self.processor.batch_decode(
            output_ids_list, skip_special_tokens=True
        )[0]
        return transcription

    def transcribe(self, audio_path):
        """Convenience method to handle complete transcription process"""
        try:
            # Preprocess audio
            input_features = self.preprocess_audio(audio_path)

            # Run encoder
            encoder_outputs = self.encode(input_features)

            # Run decoder
            output_ids = self.decode(encoder_outputs)

            # Get transcription
            transcription = self.postprocess(output_ids)

            return transcription
        except Exception as e:
            print(f"Error in transcription pipeline: {str(e)}")
            raise


def transcribe_and_rename_files(folder_path, onnx_path, q=None, max_length=250):
    # Initialize transcriber
    transcriber = WhisperONNXTranscriber(onnx_path, q=q)

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Process each file in the folder
    for dirpath, foldernames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
            else:
                continue

            # Check if the file is a .wav or .mp3
            if filename.lower().endswith(('.wav', '.mp3')):
                try:
                    # Get transcription using the convenience method
                    transcription = transcriber.transcribe(file_path)

                    # Create a valid filename from the transcription
                    valid_filename = ''.join(c for c in transcription if c.isalnum() or c in (' ', '_')).rstrip().lstrip()

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
                    print(f"Error processing {filename}: {str(e)}")
                    continue


# Example usage
folder_path = "C:/Users/MASTER/Desktop/All_fac_yuri_audio"
onnx_path = "./whisper-large-v3-turbo-onnx"  # Path to the ONNX model files
transcribe_and_rename_files(folder_path, onnx_path, q="quantized", max_length=100)
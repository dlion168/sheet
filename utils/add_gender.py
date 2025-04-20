import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from pathlib import Path
import soundfile as sf

class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

# Initialize model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

def process_batch(audio_paths: list, sampling_rate: int) -> list:
    """
    Predict gender for a batch of WAV files.
    Returns a list of gender predictions ('Female', 'Male', or 'Child') for each file in the batch.
    """
    signals = []
    valid_paths = []

    # Read and preprocess WAV files
    for audio_path in audio_paths:
        try:
            signal, sr = sf.read(audio_path)
            if sr != sampling_rate:
                raise ValueError(f"Sampling rate mismatch: expected {sampling_rate}, but got {sr}")
            
            if len(signal.shape) > 1:  # Handle stereo audio
                signal = np.mean(signal, axis=1)  # Convert to mono by averaging channels
            signal = signal.astype(np.float32)
            signals.append(signal)
            valid_paths.append(audio_path)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")

    if not signals:
        return ["Error"] * len(audio_paths)  # Return errors for all if none are valid

    # Process signals with Wav2Vec2 processor
    inputs = processor(signals, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    # Predict gender for the batch
    with torch.no_grad():
        _, _, logits_gender = model(input_values)
        gender_probabilities = logits_gender.detach().cpu().numpy()
        gender_labels = ['Female', 'Male', 'Child']
        predicted_genders = [gender_labels[np.argmax(probs)] for probs in gender_probabilities]

    # Map predictions to original order
    predictions = []
    valid_paths_set = set(valid_paths)
    for path in audio_paths:
        if path in valid_paths_set:
            predictions.append(predicted_genders[valid_paths.index(path)])
        else:
            predictions.append("Error")

    return predictions

def predict_and_update_csv(input_csv_path: str, output_csv_path: str, sampling_rate: int = 16000, batch_size: int = 8):
    """
    Read a CSV, process WAV files to predict gender in batches, and save the updated CSV.
    """
    # Load the CSV
    df = pd.read_csv(input_csv_path)
    
    # Ensure the `wav_path` column exists
    if 'wav_path' not in df.columns:
        raise ValueError("The CSV file must contain a 'wav_path' column.")
    
    # Prepare a list to store gender predictions
    gender_predictions = []

    # Process WAV files in batches
    audio_paths = df['wav_path'].tolist()
    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch_paths = audio_paths[i:i + batch_size]
        predictions = process_batch(batch_paths, sampling_rate)
        gender_predictions.extend(predictions)

    # Add the predictions as a new column
    df['gender'] = gender_predictions

    # Save the updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict gender from WAV files and update a CSV.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of the audio files (default: 16000 Hz).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing audio files.')

    args = parser.parse_args()

    # Run the prediction and CSV update
    predict_and_update_csv(args.input_csv, args.output_csv, args.sampling_rate, args.batch_size)

if __name__ == "__main__":
    main()


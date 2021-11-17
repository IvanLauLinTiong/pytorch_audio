import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os


# *classID:
# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music


class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_files, audio_dir, transformations, target_sample_rate, num_samples, device):
        super(UrbanSoundDataset, self).__init__()
        self.device = device
        self.annotations = pd.read_csv(annotation_files)
        self.audio_dir = audio_dir
        self.transformations = transformations.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path) # signals:  [num_channels,  samples]

        # Migrating signal to gpu if any
        signal = signal.to(self.device)

        # Resample signal to same sampling rate
        signal = self._resample_if_necessary(signal, sample_rate)

        # Convert stereo or more than 1 channel to mono if any
        signal = self._mix_down_if_necessary(signal)

        # Cut if number of samples in signals is more than expected
        signal = self._cut_if_necessary(signal)

        #  Padding if number of samples in signals is less than expected
        signal = self._right_pad_if_necessary(signal)

        # Apply passed-in transformations
        signal = self.transformations(signal)

        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.loc[index, 'fold']}"
        path = os.path.join(self.audio_dir, fold, self.annotations.loc[index, 'slice_file_name'])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.loc[index, 'classID']

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate).to(self.device)
            signal = resampler(signal)
            # signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=self.target_sample_rate)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        # signals -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)  # (num_pads_to_left, num_pads_to_right)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__ == "__main__":
    # ANNOTATION_FILE = ".\\dataset\\UrbanSound8K.csv"
    # AUDIO_DIR = ".\\dataset\\audio"
    ANNOTATION_FILE = "./dataset/UrbanSound8K.csv"
    AUDIO_DIR = "./dataset/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = UrbanSoundDataset(
        ANNOTATION_FILE,
        AUDIO_DIR,
        mel_spectogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )
    print(f"Samples of dataset: {len(dataset)}")

    signal, label = dataset[29]  # aka. signal shape: [1, 64, 44] [n_channels, n_mels, n_frames]
    # print(signal)

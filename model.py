import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

LABELS = ["English", "Spanish", "French", "Arabic"]
TARGET_LENGTH = 5 * 16000
N_MELS = 128

class AudioCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 19, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def fix_audio_length(audio):
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    elif len(audio) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(audio)
        audio = np.pad(audio, (0, padding))
    return audio.astype(np.float32)

def audio_to_mel(audio, sr=16000):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel = librosa.power_to_db(mel)
    return torch.tensor(mel, dtype=torch.float32)

def load_model(path="best_model.pt"):
    model = AudioCNN(n_classes=len(LABELS))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

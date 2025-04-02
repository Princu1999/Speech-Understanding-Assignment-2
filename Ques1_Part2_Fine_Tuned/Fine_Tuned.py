# Importing Important Libraries

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, accuracy_score
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score

# Initializing Dataset Paths
vox1_root = '/iitjhome/m24csa024/myenv/vox1_test_wav'
vox2_root = '/iitjhome/m24csa024/myenv/vox2_test_acc'

# Creating Custom Datasets for VoxCeleb1 and VoxCeleb2

class VoxCeleb1Dataset(Dataset):
    def __init__(self, root_dir, trial_file):
        self.root_dir = os.path.join(root_dir, "wav")
        self.trials = []
        with open(trial_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.trials.append((parts[1], parts[2], int(parts[0])))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        path1, path2, label = self.trials[idx]
        full_path1 = os.path.join(self.root_dir, path1)
        full_path2 = os.path.join(self.root_dir, path2)
        audio1 = load_audio(full_path1)
        audio2 = load_audio(full_path2)
        return audio1, audio2, label

import os
class VoxCeleb2Dataset(Dataset):
    def __init__(self, root_dir, identities, transform=None):
        self.root_dir = os.path.join(root_dir, "acc")
        self.samples = []
        self.labels = []
        self.transform = transform
        self.identity2label = {identity: idx for idx, identity in enumerate(identities)}
        for identity in identities:
            identity_folder = os.path.join(self.root_dir, identity)
            if not os.path.isdir(identity_folder):
                continue
            for root, dirs, files in os.walk(identity_folder):
                for file in files:
                    if file.endswith(".wav"):
                        self.samples.append(os.path.join(root, file))
                        self.labels.append(self.identity2label[identity])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        audio = load_audio(path)
        if self.transform:
            audio = self.transform(audio)
        return audio, label
    
# Helper Function to Load Audio Samples

def load_audio(path, sr=16000):
    """
    Load an audio file and resample if necessary.
    Return a 1D numpy array.
    """
    audio, orig_sr = librosa.load(path, sr=sr)
    return audio

# Loading the Pre-Trained WAVLM-Base-Plus Speaker Verification Mode

class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.embedding_dim = 256
        self.projection = nn.Linear(self.model.config.hidden_size, self.embedding_dim)

    def forward(self, audio):
        """
        audio: can be a 1D tensor (for a single sample) or a 2D tensor (batch, n_samples).
        We'll convert it to a list of 1D numpy arrays for the feature extractor.
        """
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                audio = audio.squeeze()
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.ndim == 1:
            audio_list = [audio_np]
        else:
            audio_list = [x for x in audio_np]

        inputs = self.feature_extractor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        embedding = self.projection(pooled)
        return F.normalize(embedding, p=2, dim=1)
    
# Defining LORA and adding it to the Mode

class LoRALinear(nn.Module):
    def __init__(self, orig_linear, r=8):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        for param in self.orig_linear.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(orig_linear.out_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, orig_linear.in_features) * 0.01)

    def forward(self, x):
        return self.orig_linear(x) + (x @ self.lora_B.t() @ self.lora_A.t())

def add_lora_to_model(model, r=8):
    model.projection = LoRALinear(model.projection, r=r)
    return model

# Defining ArcFace Loss

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=30.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, normalized_weights)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        cosine_m = cosine - one_hot * self.margin
        logits = cosine_m * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss
    
# Helper Functions to Compute EER, TAR@FAR and Speaker Identification Accuracy

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[idx]
    return eer * 100

def compute_tar_at_far(scores, labels, far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.where(fpr <= far)[0]
    if len(idx) == 0:
        return 0.0
    tar = max(tpr[idx])
    return tar * 100

def compute_identification_accuracy(pred_labels, true_labels):
    return accuracy_score(true_labels, pred_labels) * 100

# Fine-Tuning and Saving the Pre-Trained Model

def collate_fn(batch):
    audios, labels = zip(*batch)
    return list(audios), torch.tensor(labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SpeakerEmbeddingModel().to(device)

all_identities = sorted([d for d in os.listdir(os.path.join(vox2_root, "aac"))
                           if os.path.isdir(os.path.join(vox2_root, "aac", d))])
if len(all_identities) < 118:
    print("Warning: Expected at least 118 identities for VoxCeleb2, found:", len(all_identities))

train_identities = all_identities[:100]
test_identities = all_identities[100:118]

vox2_train_dataset = VoxCeleb2Dataset(root_dir=vox2_root, identities=train_identities)
train_loader = DataLoader(vox2_train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = add_lora_to_model(model, r=8)

arcface_loss_fn = ArcFaceLoss(embedding_dim=model.embedding_dim,
                              num_classes=len(train_identities),
                              margin=0.5,
                              scale=30.0).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(arcface_loss_fn.parameters()), lr=1e-4)

scaler = torch.cuda.amp.GradScaler()

num_epochs = 1
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for audio_list, labels in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}"):
        labels = labels.to(device)
        optimizer.zero_grad()
        # Use autocast for mixed precision training
        with torch.amp.autocast('cuda'):
            embeddings = model(audio_list)
            loss = arcface_loss_fn(embeddings, labels)
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
    torch.cuda.empty_cache()


# Save the fine-tuned model
finetuned_model_path = '/iitjhome/m24csa024/myenv/finetuned_speaker_model2.pth'
torch.save(model.state_dict(), finetuned_model_path)
print(f"Finetuned model saved to {finetuned_model_path}")

# Evaluating the Fine-Tuned Model on VoxCeleb1 Trial Pairs

trial_file = os.path.join(vox1_root, "trials.txt")
if not os.path.exists(trial_file):
    print("Trial file not found at", trial_file)
else:
    vox1_dataset = VoxCeleb1Dataset(root_dir=vox1_root, trial_file=trial_file)
    vox1_loader = DataLoader(vox1_dataset, batch_size=1, shuffle=False)

    post_scores = []
    post_labels = []

    with torch.no_grad():
        for audio1, audio2, label in tqdm(vox1_loader, desc="Evaluating fine-tuned model"):
            emb1 = model(audio1)
            emb2 = model(audio2)
            score = F.cosine_similarity(emb1, emb2).item()
            post_scores.append(score)
            post_labels.append(label.item())

    def compute_eer(scores, labels):
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[idx]
        return eer * 100

    def compute_tar_at_far(scores, labels, far=0.01):
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.where(fpr <= far)[0]
        if len(idx) == 0:
            return 0.0
        tar = max(tpr[idx])
        return tar * 100

    def compute_identification_accuracy(pred_labels, true_labels):
        return accuracy_score(true_labels, pred_labels) * 100

    post_eer = compute_eer(np.array(post_scores), np.array(post_labels))
    post_tar = compute_tar_at_far(np.array(post_scores), np.array(post_labels))
    post_pred_labels = [1 if s > 0.5 else 0 for s in post_scores]
    post_id_acc = compute_identification_accuracy(post_pred_labels, post_labels)

    print("\nFine-tuned Model Evaluation on VoxCeleb1:")
    print(f"EER: {post_eer:.2f}%")
    print(f"TAR@1%FAR: {post_tar:.2f}%")
    print(f"Identification Accuracy: {post_id_acc:.2f}%")

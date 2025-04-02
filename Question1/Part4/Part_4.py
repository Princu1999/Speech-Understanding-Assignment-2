import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from speechbrain.pretrained import SepformerSeparation
import pickle
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# File logging setup
log_file_path = "/iitjhome/m24csa024/myenv/SU_Q1_PART4_LOG.txt"
log_file = open(log_file_path, "a")

def log_message(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


# Load train pairs and test pairs

train_pairs_path = '/iitjhome/m24csa024/myenv/train_pairs.pkl'
test_pairs_path  = '/iitjhome/m24csa024/myenv/test_pairs.pkl'
with open(train_pairs_path, 'rb') as f:
    train_pairs = pickle.load(f)
with open(test_pairs_path, 'rb') as f:
    test_pairs = pickle.load(f)

train_speakers = list(set([pair['speaker1'] for pair in train_pairs] + [pair['speaker2'] for pair in train_pairs]))
train_speakers.sort()

rank1_accuracy_pre = 0
rank1_accuracy_fine = 0


# Loss function
def si_sdr_loss(s, s_hat, eps=1e-8):
    s = s.flatten()
    s_hat = s_hat.flatten()
    T = min(s.numel(), s_hat.numel())
    s = s[:T]
    s_hat = s_hat[:T]

    s_target = torch.dot(s_hat, s) * s / (torch.norm(s)**2 + eps)
    e_noise = s_hat - s_target
    loss = -10 * torch.log10((torch.norm(s_target)**2) / (torch.norm(e_noise)**2 + eps) + eps)
    return loss

speaker2idx = {spk: i for i, spk in enumerate(train_speakers)}
num_speakers = len(speaker2idx)


# Creating a Custom Dataset
class MultiSpeakerDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        mixed = pair['mixed']
        sources = [pair['audio1'], pair['audio2']]
        labels = [pair['speaker1'], pair['speaker2']]
        return mixed, sources, labels

train_dataset = MultiSpeakerDataset(train_pairs)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: x)

# Creating a Combined Pipeline
class CombinedPipeline(nn.Module):
    def __init__(self, sepformer, speaker_embedder, num_classes):
        super().__init__()
        self.sepformer = sepformer
        self.speaker_embedder = speaker_embedder
        for param in self.speaker_embedder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, mixed_audio):
        sep_out = self.sepformer.separate_batch(mixed_audio)
        batch_size, n_src, _ = sep_out.shape
        logits_list = []
        enhanced_list = []

        for i in range(n_src):
            src = sep_out[:, i, :]
            embeddings = []
            for j in range(batch_size):
                audio = src[j].detach()

                if audio.shape[0] < 4096:
                    pad_len = 4096 - audio.shape[0]
                    audio = F.pad(audio, (0, pad_len), "constant", 0)

                with torch.no_grad():
                    emb = self.speaker_embedder(audio)
                embeddings.append(emb)
            embeddings = torch.cat(embeddings, dim=0)
            logits = self.classifier(embeddings)
            logits_list.append(logits)
            enhanced_list.append(src)

        logits_out = torch.stack(logits_list, dim=1)
        enhanced_out = torch.stack(enhanced_list, dim=1)
        return enhanced_out, logits_out

sepformer = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sepformer = sepformer.to(device)
sepformer.device = device 

if hasattr(sepformer, "mods"):
    for key, mod in sepformer.mods.items():
        sepformer.mods[key] = mod.to(device)
        

# Loading the Wavlm-Base-Plus Pre-Trained Model
class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.embedding_dim = 256
        self.projection = nn.Linear(self.model.config.hidden_size, self.embedding_dim)

    def forward(self, audio):
        if isinstance(audio, list):
            audio_list = audio
        elif isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                audio = audio.squeeze()
            audio_np = audio.cpu().numpy()
            if audio_np.ndim == 1:
                audio_list = [audio_np]
            else:
                audio_list = [x for x in audio_np]
        else:
            if hasattr(audio, "ndim") and audio.ndim == 1:
                audio_list = [audio]
            else:
                audio_list = [x for x in audio]
        inputs = self.feature_extractor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        embedding = self.projection(pooled)
        return F.normalize(embedding, p=2, dim=1)
    
    
# Defining and adding LORA to the model
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
        lora_A = self.lora_A.to(x.device)
        lora_B = self.lora_B.to(x.device)
        return self.orig_linear(x) + (x @ lora_B.t() @ lora_A.t())

def add_lora_to_model(model, r=8):
    model.projection = LoRALinear(model.projection, r=r)
    return model

pretrained_model_path = '/iitjhome/m24csa024/myenv/pretrained_speaker_model.pth'
finetuned_model_path  = '/iitjhome/m24csa024/myenv/finetuned_speaker_model.pth'

pretrained_model = SpeakerEmbeddingModel().to(device)
pretrained_state = torch.load(pretrained_model_path, map_location=device)
pretrained_model.load_state_dict(pretrained_state)

combined_model = CombinedPipeline(sepformer, pretrained_model, num_speakers).to(device)
combined_model.train()

optimizer = torch.optim.Adam(combined_model.parameters(), lr=1e-4)
classification_loss_fn = nn.CrossEntropyLoss()

num_epochs = 5
log_message("Starting training of the combined pipeline...")

scaler = torch.amp.GradScaler('cuda') 

# Training the model on Train Pairs
for epoch in range(num_epochs):
    epoch_sep_loss = 0.0
    epoch_cls_loss = 0.0
    count = 0
    for batch in train_loader:
        batch_loss = 0.0
        for (mixed, sources, labels) in batch:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda'):
                mixed_tensor = torch.tensor(mixed, dtype=torch.float32, device=device).unsqueeze(0)
                gt_sources = []
                for src in sources:
                    gt_sources.append(torch.tensor(src, dtype=torch.float32, device=device).unsqueeze(0))
                gt_sources = torch.stack(gt_sources, dim=1)
                
                enhanced, logits = combined_model(mixed_tensor)
                n_src = enhanced.shape[1]
                n_src_gt = gt_sources.shape[1]
                n_src = min(n_src, n_src_gt)
                sep_loss = 0.0
                for i in range(n_src):
                    est = enhanced[0, i, :]
                    gt = gt_sources[0, i, :]
                    sep_loss += si_sdr_loss(gt, est)
                sep_loss = sep_loss / n_src
                
                n_src_est = enhanced.shape[1]
                n_src_labels = len(labels)
                n_src = min(n_src_est, n_src_labels)
                cls_loss = 0.0
                for i in range(n_src):
                    logit = logits[0, i, :]
                    gt_label = speaker2idx[labels[i]] if labels[i] in speaker2idx else 0
                    cls_loss += classification_loss_fn(logit.unsqueeze(0), torch.tensor([gt_label], device=device))
                cls_loss = cls_loss / n_src
                
                total_loss = sep_loss + cls_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss += total_loss.item()
            epoch_sep_loss += sep_loss.item()
            epoch_cls_loss += cls_loss.item()
            count += 1
        msg = f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {batch_loss/len(batch):.4f}"
        log_message(msg)
    msg = f"Epoch [{epoch+1}/{num_epochs}] Avg Sep Loss: {epoch_sep_loss/count:.4f}"
    log_message(msg)

log_message("Training completed. Setting model to eval mode...")
combined_model.eval()

log_message("Training completed. Evaluating on test set...")

sdr_list = []
sir_list = []
sar_list = []
pesq_list = []
correct_cls = 0
total_cls = 0

idx2speaker = {v: k for k, v in speaker2idx.items()}

# Evaluating the model on Test Pairs
with torch.no_grad():
    for pair in test_pairs:
        mixed_tensor = torch.tensor(pair['mixed'], dtype=torch.float32, device=device).unsqueeze(0)
        enhanced, logits = combined_model(mixed_tensor)
        enhanced_np = enhanced.cpu().numpy()[0]
        if enhanced_np.shape[0] != 2:
            enhanced_np = enhanced_np.T
        references = np.stack([pair['audio1'], pair['audio2']], axis=0)
        sdr, sir, sar, perm = bss_eval_sources(references, enhanced_np)
        sdr_list.append(np.mean(sdr))
        sir_list.append(np.mean(sir))
        sar_list.append(np.mean(sar))
        pesq_scores = []
        for i in range(2):
            try:
                score = pesq(16000, references[i], enhanced_np[perm[i]], 'wb')
            except Exception as e:
                score = np.nan
            pesq_scores.append(score)
        pesq_list.append(np.nanmean(pesq_scores))
        
        idx2speaker = {v: k for k, v in speaker2idx.items()}
        for i in range(2):
            pred_class = torch.argmax(logits[0, i, :]).item()
            gt = pair['speaker1'] if i == 0 else pair['speaker2']
            if idx2speaker.get(pred_class, None) == gt:
                correct_cls += 1
            total_cls += 1

avg_sdr = np.nanmean(sdr_list)
avg_sir = np.nanmean(sir_list)
avg_sar = np.nanmean(sar_list)
avg_pesq = np.nanmean(pesq_list)
rank1_accuracy = correct_cls / total_cls * 100 if total_cls > 0 else 0

log_message("\nEvaluation on Test Set:")
log_message(f"Separation Metrics -> SDR: {avg_sdr:.2f} dB, SIR: {avg_sir:.2f} dB, SAR: {avg_sar:.2f} dB, PESQ: {avg_pesq:.2f}")
log_message(f"Speaker Identification Rank-1 Accuracy (Combined Model): {rank1_accuracy:.2f}%")
log_message(f"Speaker Identification Rank-1 Accuracy (Pre-Trained Model): {rank1_accuracy_pre:.2f}%")
log_message(f"Speaker Identification Rank-1 Accuracy (Fine-Tuned Model): {rank1_accuracy_fine:.2f}%")

log_file.close()

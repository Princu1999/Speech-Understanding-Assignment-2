# Suppressing Warnings related to Various Packages that arrised due to version mismatches
 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message="Deprecated as of librosa version 0.10.0.")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.amp")

# Importing Important Libraries

import os
import glob
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.pretrained import SepformerSeparation
from mir_eval.separation import bss_eval_sources
from pesq import pesq
import pickle


from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Helper function for creating a multi-speaker scenario dataset by mixing/overlapping utterances from 2 different speakers


vox2_audio_path = '/iitjhome/m24csa024/myenv/vox2-test-aac/aac'
vox2_txt_path   = '/iitjhome/m24csa024/myenv/vox2-test-txt/txt'


all_speakers = sorted(os.listdir(vox2_txt_path))

# Using first 50 audio samples for training and next 50 audio samples for testing.
train_speakers = all_speakers[:50]
test_speakers  = all_speakers[50:100]

print(f"Total speakers: {len(all_speakers)}; Training speakers: {len(train_speakers)}; Testing speakers: {len(test_speakers)}")

def load_random_audio(speaker_folder, dataset_path, sr=16000):
    """
    Given a speaker folder and base dataset path, randomly choose one subfolder and one .m4a file.
    """
    speaker_dir = os.path.join(dataset_path, speaker_folder)
    subfolders = os.listdir(speaker_dir)
    if len(subfolders) == 0:
        return None
    chosen_subfolder = random.choice(subfolders).
    file_candidates = glob.glob(os.path.join(speaker_dir, chosen_subfolder, '*.m4a'))
    if not file_candidates:
        return None
    chosen_file = random.choice(file_candidates)
    try:
        audio, _ = librosa.load(chosen_file, sr=sr)
    except Exception as e:
        print(f"Error loading {chosen_file}: {e}")
        return None
    return audio

def mix_audios(audio1, audio2):
    """
    Pads two audio signals to the same length and sums them to create a mixed signal.
    """
    length = max(len(audio1), len(audio2))
    audio1 = np.pad(audio1, (0, length - len(audio1)), mode='constant')
    audio2 = np.pad(audio2, (0, length - len(audio2)), mode='constant')
    mixed  = audio1 + audio2
    return audio1, audio2, mixed


num_train_pairs = 100  # Adjust as needed.
train_pairs = []
for _ in range(num_train_pairs):
    spk1, spk2 = random.sample(train_speakers, 2)
    audio1 = load_random_audio(spk1, vox2_audio_path)
    audio2 = load_random_audio(spk2, vox2_audio_path)
    if audio1 is None or audio2 is None:
        continue
    ref1, ref2, mixed = mix_audios(audio1, audio2)
    train_pairs.append({
        'speaker1': spk1,
        'speaker2': spk2,
        'audio1': ref1,
        'audio2': ref2,
        'mixed': mixed
    })

num_test_pairs = 50  # Adjust as needed.
test_pairs = []
for _ in range(num_test_pairs):
    spk1, spk2 = random.sample(test_speakers, 2)
    audio1 = load_random_audio(spk1, vox2_audio_path)
    audio2 = load_random_audio(spk2, vox2_audio_path)
    if audio1 is None or audio2 is None:
        continue
    ref1, ref2, mixed = mix_audios(audio1, audio2)
    test_pairs.append({
        'speaker1': spk1,
        'speaker2': spk2,
        'audio1': ref1,
        'audio2': ref2,
        'mixed': mixed
    })

print(f"Created {len(train_pairs)} training pairs and {len(test_pairs)} testing pairs.")


output_dir = '/iitjhome/m24csa024/myenv'
train_file_path = os.path.join(output_dir, "train_pairs.pkl")
test_file_path  = os.path.join(output_dir, "test_pairs.pkl")

with open(train_file_path, "wb") as f:
    pickle.dump(train_pairs, f)
with open(test_file_path, "wb") as f:
    pickle.dump(test_pairs, f)

print(f"Train pairs saved to {train_file_path}")
print(f"Test pairs saved to {test_file_path}")

# Loading the Pre-Trained WAVLM-Base-Plus Speaker Verification Model

class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.embedding_dim = 256
        self.projection = nn.Linear(self.model.config.hidden_size, self.embedding_dim)

    def forward(self, audio):
        # audio: list of numpy arrays or torch tensor.
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
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)
        pooled = hidden_states.mean(dim=1)  # mean pooling over time
        embedding = self.projection(pooled)
        return F.normalize(embedding, p=2, dim=1)

# Loading the Pre-Trained WAVLM-Base-Plus Speaker Verification Model

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

# Helper Function to Compute Cosine Similarity

def identify_speaker(embedding, ref_embeddings):
    """
    Given an embedding and a dictionary of reference embeddings,
    compute cosine similarities and return the speaker ID with highest similarity.
    """
    similarities = {}
    for spk, ref_emb in ref_embeddings.items():
        sim = torch.cosine_similarity(embedding, ref_emb, dim=1)
        similarities[spk] = sim.item()
    identified = max(similarities, key=similarities.get)
    return identified

# Initializing models and loading state_dict

pretrained_model_path = '/iitjhome/m24csa024/myenv/pretrained_speaker_model.pth'
finetuned_model_path  = '/iitjhome/m24csa024/myenv/finetuned_speaker_model.pth'

# Initializing models and loading state_dict
pretrained_model = SpeakerEmbeddingModel().to(device)
pretrained_state = torch.load(pretrained_model_path, map_location=device)
pretrained_model.load_state_dict(pretrained_state)

finetuned_model = SpeakerEmbeddingModel().to(device)
finetuned_model = add_lora_to_model(finetuned_model, r=8)
finetuned_state = torch.load(finetuned_model_path, map_location=device)
finetuned_model.load_state_dict(finetuned_state)

# Various Metrices for Question1 Part III (A & B)

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
sepformer.eval()
print("Loaded pre-trained SepFormer for separation.")


sdr_list = []
sir_list = []
sar_list = []
pesq_list = []

for pair in test_pairs:
    mixed_tensor = torch.tensor(pair['mixed'], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        est_sources = sepformer.separate_batch(mixed_tensor)
    est_sources = est_sources.cpu().numpy()[0]
    if est_sources.shape[0] != 2:
        est_sources = est_sources.T
    references = np.stack([pair['audio1'], pair['audio2']], axis=0)
    sdr, sir, sar, perm = bss_eval_sources(references, est_sources)
    sdr_list.append(np.mean(sdr))
    sir_list.append(np.mean(sir))
    sar_list.append(np.mean(sar))
    
    pesq_scores = []
    for i in range(2):
        try:
            score = pesq(16000, references[i], est_sources[perm[i]], 'wb')
        except Exception as e:
            score = np.nan
        pesq_scores.append(score)
    pesq_list.append(np.nanmean(pesq_scores))

avg_sdr  = np.nanmean(sdr_list)
avg_sir  = np.nanmean(sir_list)
avg_sar  = np.nanmean(sar_list)
avg_pesq = np.nanmean(pesq_list)

print("\nSeparation Metrics on Test Set:")
print(f"SDR: {avg_sdr:.2f} dB")
print(f"SIR: {avg_sir:.2f} dB")
print(f"SAR: {avg_sar:.2f} dB")
print(f"PESQ: {avg_pesq:.2f}")

pretrained_model.eval()
finetuned_model.eval()
print("Loaded speaker identification models.")

reference_embeddings = {}
for spk in test_speakers:
    ref_audio = load_random_audio(spk, vox2_audio_path)
    if ref_audio is None:
        continue
    ref_audio_tensor = torch.tensor(ref_audio, dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = pretrained_model(ref_audio_tensor)
    reference_embeddings[spk] = emb

print(f"Created reference embeddings for {len(reference_embeddings)} test speakers.")


MIN_AUDIO_LENGTH = 10 
correct_pretrained = 0
correct_finetuned  = 0
total = 0

for pair in test_pairs:
    mixed_tensor = torch.tensor(pair['mixed'], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        est_sources = sepformer.separate_batch(mixed_tensor)
    est_sources = est_sources.cpu().numpy()[0]

    if est_sources.shape[0] != 2:
        est_sources = est_sources.T
    for i in range(2):
        est_audio = est_sources[i]
        if len(est_audio) < MIN_AUDIO_LENGTH:

            est_audio = np.pad(est_audio, (0, max(0, MIN_AUDIO_LENGTH - len(est_audio))), 
                              mode='constant')

        est_audio_tensor = torch.tensor(est_audio, dtype=torch.float32).to(device)
        if len(est_audio) < MIN_AUDIO_LENGTH:
            continue
        with torch.no_grad():
            try:
                emb_pre  = pretrained_model(est_audio_tensor)
                emb_fine = finetuned_model(est_audio_tensor)
            except RuntimeError as e:
                print(f"Skipping short audio: {e}")
                continue
        identified_pre  = identify_speaker(emb_pre, reference_embeddings)
        identified_fine = identify_speaker(emb_fine, reference_embeddings)

        gt = pair['speaker1'] if i == 0 else pair['speaker2']
        if identified_pre == gt:
            correct_pretrained += 1
        if identified_fine == gt:
            correct_finetuned += 1
        total += 1

rank1_pretrained = correct_pretrained / total * 100 if total > 0 else 0
rank1_finetuned  = correct_finetuned / total * 100 if total > 0 else 0

print("\nSpeaker Identification Rank-1 Accuracy:")
print(f"Pre-trained model: {rank1_pretrained:.2f}%")
print(f"Fine-tuned model: {rank1_finetuned:.2f}%")

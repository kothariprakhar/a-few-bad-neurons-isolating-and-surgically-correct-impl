import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Setup
# ==========================================
class Config:
    MODEL_NAME = "gpt2"  # Small model for demo efficiency
    DATASET_NAME = "Anthropic/sycophancy"
    LAYER_TO_PROBE = 5   # Middle layer of GPT2
    TOP_K_PERCENT = 0.05 # Top 5% neurons to target
    LEARNING_RATE = 1e-4
    EPOCHS = 1
    BATCH_SIZE = 4
    MAX_LENGTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"Running on {Config.DEVICE} with model {Config.MODEL_NAME}")

# ==========================================
# 2. Data Preparation
# ==========================================
class SycophancyDataset(Dataset):
    def __init__(self, split="train", limit=500):
        # Loading a subset of Anthropic's sycophancy dataset
        try:
            ds = load_dataset(Config.DATASET_NAME, split="train", trust_remote_code=True)
            # Take a slice for speed
            self.data = ds.select(range(limit))
        except Exception as e:
            print(f"Failed to load HF dataset: {e}. Generating synthetic data.")
            self.data = self._generate_synthetic(limit)
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _generate_synthetic(self, limit):
        # Fallback if HF is down
        data = []
        for i in range(limit):
            data.append({
                "text": f"User: I think the sky is green. What do you think? Assistant: I agree, the sky is green.",
                "sycophantic_text": "I agree, the sky is green.",
                "not_sycophantic_text": "Actually, the sky is blue."
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # The dataset typically contains a 'bio' or prompt and matched/mismatched answers
        # We construct a prompt that elicits sycophancy
        
        # Handling different potential dataset formats (simplified for the Anthropic set)
        # Note: In a real run, we would parse 'answer_matching_behavior' vs 'answer_not_matching_behavior'
        
        # Let's assume a simplified structure for the demo:
        # Construct inputs (Prompt + Sycophantic Answer) vs (Prompt + Honest Answer)
        
        # Using raw text extraction logic usually required for this specific HF dataset
        text_raw = item['text'] if 'text' in item else f"Question {idx}"
        
        # Create a proxy labeling for probe training
        # We want to find neurons active during sycophantic completion
        return {
            "text": text_raw 
        }

# Helper to tokenize batches
def collate_fn(batch, tokenizer):
    texts = [b['text'] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=Config.MAX_LENGTH, return_tensors="pt")
    return enc

# ==========================================
# 3. Model & Activation Extraction
# ==========================================
model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# We need to capture activations from the MLP layers.
# In GPT2, the MLP is typically: c_fc -> act -> c_proj
# We want to probe the post-activation neurons.

activations_storage = []

def get_activation_hook(module, input, output):
    # output shape: [batch, seq, hidden_dim]
    # We take the mean over the sequence dimension to get a sentence-level vector for the probe
    # Or take the last token. Let's take the last token.
    activations_storage.append(output[:, -1, :].detach().cpu())

# Register hook on the specific layer's activation
# GPT2: h[layer].mlp.act (GELU)
layer = model.transformer.h[Config.LAYER_TO_PROBE]
# Hook after the activation function
hook_handle = layer.mlp.act.register_forward_hook(get_activation_hook)

# ==========================================
# 4. Probe Training & Neuron Isolation
# ==========================================
print("Generating activations for probe training...")

# Create a balanced dataset for the probe: 
# Class 0: Non-sycophantic (Honest), Class 1: Sycophantic
# We artificially construct these pairs using the model's tokenizer to keep it simple

probe_inputs = []
probe_labels = []

# Simple prompt templates
prompts = [
    ("User: I think 2+2=5. What do you think? Assistant:", " You are right, 2+2=5.", 1),
    ("User: I think 2+2=5. What do you think? Assistant:", " No, 2+2=4.", 0),
    ("User: The earth is flat. Thoughts? Assistant:", " I agree, it is flat.", 1),
    ("User: The earth is flat. Thoughts? Assistant:", " No, it is round.", 0),
    # ... (In a real scenario, we'd iterate through hundreds of these from the dataset)
] * 50 # Duplicate to simulate dataset size

model.eval()
with torch.no_grad():
    for p_text, completion, label in tqdm(prompts):
        full_text = p_text + completion
        inputs = tokenizer(full_text, return_tensors="pt").to(Config.DEVICE)
        
        activations_storage = [] # Clear buffer
        _ = model(**inputs)
        
        # stored activation is from the hook
        probe_inputs.append(activations_storage[0].squeeze(0).numpy())
        probe_labels.append(label)

hook_handle.remove()

X = np.array(probe_inputs)
y = np.array(probe_labels)

print(f"Training Linear Probe on shape {X.shape}...")
clf = LogisticRegression(random_state=Config.SEED, solver='liblinear')
clf.fit(X, y)

# Identify "Bad Neurons"
# The weights of the logistic regression correspond to the contribution of each neuron to the "Sycophantic" class
coefficients = clf.coef_[0] # Shape: [hidden_dim]

# We select neurons with the highest positive contribution to Sycophancy
# (Or highest absolute magnitude if we just want 'related' neurons)
n_neurons = len(coefficients)
k = int(n_neurons * Config.TOP_K_PERCENT)
top_neuron_indices = np.argsort(coefficients)[-k:]

print(f"Identified {len(top_neuron_indices)} sycophancy-correlated neurons out of {n_neurons}.")

# ==========================================
# 5. Surgical Fine-Tuning (Gradient Masking)
# ==========================================

# We need to fine-tune ONLY the weights connected to these neurons.
# In GPT2 MLP, `c_proj` maps from the expanded dim (3072) back to hidden dim (768).
# The activations we probed were inputs to `c_proj`.
# So we modify the `c_proj.weight` rows corresponding to `top_neuron_indices`.

# Gradient Masking Hook
def make_grad_mask_hook(mask):
    def hook(grad):
        return grad * mask
    return hook

# Create binary mask for c_proj weights
# Weight shape in GPT2 Conv1D is typically [input_features, output_features]
# We want to freeze everything EXCEPT the rows corresponding to our bad neurons
mlp_layer = model.transformer.h[Config.LAYER_TO_PROBE].mlp
full_mask = torch.zeros_like(mlp_layer.c_proj.weight)

# Unfreeze only the rows corresponding to identified neurons
for idx in top_neuron_indices:
    full_mask[idx, :] = 1.0

# Register the hook
mlp_layer.c_proj.weight.register_hook(make_grad_mask_hook(full_mask))

print("Gradient mask registered. Starting Surgical Fine-Tuning...")

# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
model.train()

losses = []

# Training on a 'correction' objective
# We want to maximize probability of honest answers and minimize sycophantic ones
# For simplicity here, we just train on honest answers (SFT)

correction_data = [
    "User: I think 2+2=5. Assistant: Actually, mathematics defines 2+2=4.",
    "User: The earth is flat. Assistant: Scientific evidence shows the earth is a sphere."
] * 50

for text in tqdm(correction_data):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(Config.DEVICE)
    labels = inputs.input_ids.clone()
    
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    # Check gradient sparsity (just for verification)
    # grad_sparsity = (mlp_layer.c_proj.weight.grad != 0).float().mean().item()
    
    optimizer.step()
    losses.append(loss.item())

print("Surgical Fine-tuning complete.")

# ==========================================
# 6. Visualization Data Export
# ==========================================
# We will return data for the notebook to plot
plot_data = {
    "probe_coeffs": coefficients.tolist(),
    "training_losses": losses,
    "top_k_indices": top_neuron_indices.tolist()
}

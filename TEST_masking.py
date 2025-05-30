import torch
import torch.nn as nn

# Fake input: batch of 2 images, each with 5 tokens, embedding dim = 4
B, N, D = 3, 5, 4
torch.manual_seed(0)  # for reproducibility

# Simulated token embeddings (random values)
out = torch.randn(B, N, D)
print("Original token tensor (out):\n", out, "\n")

# Set masking parameters
mask_ratio = 0.4
num_mask = int(mask_ratio * N)

# Generate random indices to mask (per sample)
mask_indices = torch.rand(B, N).argsort(dim=1)[:, :num_mask]
print("Random mask indices:\n", mask_indices, "\n")

# Create mask: initially all False
mask = torch.zeros(B, N, dtype=torch.bool)
mask.scatter_(1, mask_indices, 1)  # Set True at masked positions
print("Binary mask [B, N]:\n", mask, "\n")

# Expand to [B, N, D] to match token shape
mask = mask.unsqueeze(-1).expand(-1, -1, D)

# Create a learnable mask token (simulate pre-training value)
mask_token = nn.Parameter(torch.full((1, D), 99.0))  # fixed value for clarity
masked_out = torch.where(mask, mask_token.expand(B, N, D), out)

print("Masked token tensor (after applying mask_token):\n", masked_out)
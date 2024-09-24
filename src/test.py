import torch
import torch.nn as nn

# Define dimensions
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
seq_len = 10    # Sequence length
batch_size = 4  # Batch size

# Create some random inputs
query = torch.randn(seq_len, batch_size, embed_dim)
key = torch.randn(seq_len+1, batch_size, embed_dim)
value = torch.randn(seq_len+1, batch_size, embed_dim)

# Create key_padding_mask for attention
key_padding_mask = torch.randint(0, 2, (batch_size, seq_len+1)).bool()

# Initialize multihead attention layer with batch_first=False
multihead_attn_no_batch_first = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
# Initialize multihead attention layer with batch_first=True
multihead_attn_batch_first = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

# 1. Run with batch_first=False, average_attn_weights=True
attn_output_no_batch_first, attn_weights_no_batch_first = multihead_attn_no_batch_first(
    query=query,
    key=key,
    value=value,
    key_padding_mask=key_padding_mask,
    need_weights=True,
    average_attn_weights=True
)

# 2. Run with batch_first=True, average_attn_weights=True
query_bf = query.transpose(0, 1)  # Transpose to shape (batch_size, seq_len, embed_dim)
key_bf = key.transpose(0, 1)
value_bf = value.transpose(0, 1)

attn_output_batch_first, attn_weights_batch_first = multihead_attn_batch_first(
    query=query_bf,
    key=key_bf,
    value=value_bf,
    key_padding_mask=key_padding_mask,
    need_weights=True,
    average_attn_weights=True
)

# 3. Run with batch_first=False, average_attn_weights=False
attn_output_no_batch_first_no_avg, attn_weights_no_batch_first_no_avg = multihead_attn_no_batch_first(
    query=query,
    key=key,
    value=value,
    key_padding_mask=key_padding_mask,
    need_weights=True,
    average_attn_weights=False
)

# 4. Run with batch_first=True, average_attn_weights=False
attn_output_batch_first_no_avg, attn_weights_batch_first_no_avg = multihead_attn_batch_first(
    query=query_bf,
    key=key_bf,
    value=value_bf,
    key_padding_mask=key_padding_mask,
    need_weights=True,
    average_attn_weights=False
)

# Print the shapes to compare attention outputs and weights
print("=== Batch First=False, average_attn_weights=True ===")
print("Attn Output shape (no batch first):", attn_output_no_batch_first.shape)
print("Attn Weights shape (no batch first):", attn_weights_no_batch_first.shape)

print("\n=== Batch First=True, average_attn_weights=True ===")
print("Attn Output shape (batch first):", attn_output_batch_first.shape)
print("Attn Weights shape (batch first):", attn_weights_batch_first.shape)

print("\n=== Batch First=False, average_attn_weights=False ===")
print("Attn Output shape (no batch first, no avg weights):", attn_output_no_batch_first_no_avg.shape)
print("Attn Weights shape (no batch first, no avg weights):", attn_weights_no_batch_first_no_avg.shape)

print("\n=== Batch First=True, average_attn_weights=False ===")
print("Attn Output shape (batch first, no avg weights):", attn_output_batch_first_no_avg.shape)
print("Attn Weights shape (batch first, no avg weights):", attn_weights_batch_first_no_avg.shape)

# Check if the weights are in the same order by comparing them
print("\n=== Weights Comparison (average_attn_weights=True) ===")
print(torch.allclose(attn_weights_no_batch_first, attn_weights_batch_first))

print("\n=== Weights Comparison (average_attn_weights=False) ===")
print(torch.allclose(attn_weights_no_batch_first_no_avg, attn_weights_batch_first_no_avg))

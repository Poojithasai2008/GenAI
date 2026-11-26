
import torch
import math
import torch.nn as nn

# ==========================================
# PART 1: SCALED DOT-PRODUCT ATTENTION
# ==========================================

print("=" * 50)
print("PART 1: SCALED DOT-PRODUCT ATTENTION")
print("=" * 50)

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0]
])

scores = torch.matmul(Q, K.T)
d_k = K.shape[1]

scaled_scores = scores / math.sqrt(d_k)
attention_weights = torch.softmax(scaled_scores, dim=-1)
output = torch.matmul(attention_weights, V)

print("\nQuery Matrix (Q):\n", Q)
print("\nKey Matrix (K):\n", K)
print("\nValue Matrix (V):\n", V)
print("\nK Transpose:\n", K.T)
print("\nAttention Scores:\n", scores)
print("\nScaled Scores:\n", scaled_scores)
print("\nSoftmax Weights:\n", attention_weights)
print("\nFinal Attention Output:\n", output)


# ==========================================
# PART 2: MULTI-HEAD ATTENTION
# ==========================================

print("\n" + "=" * 50)
print("PART 2: MULTI-HEAD ATTENTION")
print("=" * 50)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Identity matrices for easy manual calculations
        with torch.no_grad():
            identity = torch.eye(d_model)

            self.W_q.weight.copy_(identity)
            self.W_k.weight.copy_(identity)
            self.W_v.weight.copy_(identity)
            self.W_o.weight.copy_(identity)

    def forward(self, x):

        batch_size, seq_length, _ = x.shape

        # Step 1: Project input into Q, K and V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        print("\nProjected Q:\n", Q)
        print("\nProjected K:\n", K)
        print("\nProjected V:\n", V)

        # Step 2: Split into multiple heads
        Q = Q.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        print("\nHead 1 Q:\n", Q[0, 0])
        print("\nHead 2 Q:\n", Q[0, 1])

        # Step 3: Calculate attention scores
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )

        print("\nAttention Scores:\n", scores)

        # Step 4: Scale scores
        scaled_scores = scores / math.sqrt(self.head_dim)

        print("\nScaled Scores:\n", scaled_scores)

        # Step 5: Apply Softmax
        attention_weights = torch.softmax(
            scaled_scores, dim=-1
        )

        print("\nSoftmax Weights:\n", attention_weights)

        # Step 6: Multiply attention weights by V
        head_outputs = torch.matmul(attention_weights, V)

        print("\nHead 1 Output:\n", head_outputs[0, 0])
        print("\nHead 2 Output:\n", head_outputs[0, 1])

        # Step 7: Concatenate the heads
        concatenated = head_outputs.transpose(
            1, 2
        ).contiguous().reshape(
            batch_size, seq_length, self.d_model
        )

        print("\nConcatenated Output:\n", concatenated)

        # Step 8: Final linear projection
        final_output = self.W_o(concatenated)

        return final_output


# Input matrix
X = torch.tensor([[
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 0.0]
]])

print("\nInput Matrix X:\n", X)

# Create model with 2 attention heads
model = MultiHeadAttention(
    d_model=4,
    num_heads=2
)

# Run Multi-Head Attention
multi_head_output = model(X)

print("\nFINAL MULTI-HEAD ATTENTION OUTPUT:")
print(multi_head_output)

print("\nProgram completed successfully!")

# Positional Encoding Module 
import torch
from torch import nn
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length = 512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim # e.g., embedding_dim = 768
        self.dropout = nn.Dropout(dropout) # Dropout layer to prevent overfitting
        pe = torch.zeros(max_seq_length, embedding_dim) # Initialize positional encoding matrix with zeros
        
        for pos in range(max_seq_length): # Iterate over each position
            for i in range(0, embedding_dim, 2): # Iterate over each dimension
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim))) # Calculate sine for even indices
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim))) # Calculate cosine for odd indices
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # Register pe as a buffer to avoid it being considered a model parameter

    def forward(self, x): # x shape: (batch_size, seq_length, embedding_dim)
        x = x * math.sqrt(self.embedding_dim) # Scale input embeddings
        seq_length = x.size(1) # Get the sequence length from input
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device) # Get positional encodings for the input sequence length and move to the same device as x
        # Add positional encodings vector to the embedding vectr
        x = x + pe # Add positional encodings to input embeddings
        x = self.dropout(x) # Apply dropout
        return x
    
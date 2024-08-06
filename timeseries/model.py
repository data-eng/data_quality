import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the Multi-Head Attention module.

        :param d_model: dimension of the input and output features
        :param num_heads: number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def attention_scores(self, Q, K, V, mask=None):
        """
        Calculates the attention scores and applies them to the values.
        
        :param Q: query matrix
        :param K: key matrix
        :param V: value matrix
        :param mask: optional mask to prevent attention to certain positions
        :return: attention scores per head
        """
        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        scaling_factor = math.sqrt(self.d_k)
        attn_scores = dot_product / scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_scores = torch.matmul(attn_probs, V)

        return attn_scores
        
    def split_heads(self, x):
        """
        Split the input into multiple heads.

        :param x: tensor (batch_size, seq_length, d_model)
        :return: tensor (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.size()

        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        
        return x
        
    def combine_heads(self, x):
        """
        Combine multiple heads into a single tensor.

        :param x: tensor (batch_size, num_heads, seq_length, d_k)
        :return: tensor (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, _ = x.size()

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_length, self.d_model)
        
        return x
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for multi-head attention.

        :param Q: query matrix
        :param K: key matrix
        :param V: value matrix
        :param mask: optional mask to avoid attention on certain positions
        :return: multi-head attention matrix
        """
        Q = self.split_heads(x=self.W_q(Q))
        K = self.split_heads(x=self.W_k(K))
        V = self.split_heads(x=self.W_v(V))
        
        attn_scores = self.attention_scores(Q, K, V, mask)
        attn_matrix = self.combine_heads(attn_scores)
        
        output = self.W_o(attn_matrix)

        return output

class Transformer(nn.Module):
    def __init__(self, in_size=3, out_size=5, nhead=1, num_layers=1, dim_feedforward=2048, dropout=0):
        super(Transformer, self).__init__()

        self.nhead = nhead
        self.encoder = ...
        self.decoder = ...
        self.classifier = nn.Linear(in_size, out_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights and biases of the classifier linear layer:

        - Sets the bias of the classifier linear layer to zero.
        - Initializes the weights with values drawn from a Xavier uniform distribution.
        """ 
        self.classifier.bias.data.zero_()
        nn.init.xavier_uniform_(self.classifier.weight.data)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the transformer model.

        :param x: input tensor
        :return: output tensor
        """
        
        x = self.encoder(src=x)
        x = self.decoder(tgt=x)

        x = self.classifier(input=x)

        return x
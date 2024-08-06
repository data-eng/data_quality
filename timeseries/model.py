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
        
    def attention_scores(self, Q, K, V):
        """
        Calculates the attention scores and applies them to the values.
        
        :param Q: query matrix
        :param K: key matrix
        :param V: value matrix
        :return: attention scores per head
        """
        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        scaling_factor = math.sqrt(self.d_k)
        attn_scores = dot_product / scaling_factor

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
        
    def forward(self, Q, K, V):
        """
        Forward pass for multi-head attention.

        :param Q: query matrix
        :param K: key matrix
        :param V: value matrix
        :return: multi-head attention matrix
        """
        Q = self.split_heads(x=self.W_q(Q))
        K = self.split_heads(x=self.W_k(K))
        V = self.split_heads(x=self.W_v(V))
        
        attn_scores = self.attention_scores(Q, K, V)
        attn_matrix = self.combine_heads(attn_scores)
        
        output = self.W_o(attn_matrix)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        """
        Initialize the FeedForward module.

        :param d_model: dimension of the input and output features
        :param dim_feedforward: dimension of the feedforward network hidden layer
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the feedforward network.

        :param x: tensor (batch_size, seq_length, d_model)
        :return: tensor (batch_size, seq_length, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        """
        Initialize the Encoder module.

        :param d_model: dimension of the input and output features
        :param num_heads: number of attention heads
        :param dim_feedforward: dimension of the feedforward network hidden layer
        :param dropout: rate for randomly deactivating neurons during training
        """
        super(Encoder, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = FeedForward(d_model, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Forward pass through the encoder.

        :param src: tensor (batch_size, seq_length, d_model)
        :return: tensor (batch_size, seq_length, d_model)
        """
        self_attn_x = self.self_attn(src, src, src)

        x = x + self.dropout(self_attn_x)
        x = self.norm1(x)
        
        ffn_x = self.ffn(x)

        x = x + self.dropout(ffn_x)
        x = self.norm2(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        """
        Initialize the Decoder module.

        :param d_model: dimension of the input and output features
        :param num_heads: number of attention heads
        :param dim_feedforward: dimension of the feedforward network hidden layer
        :param dropout: rate for randomly deactivating neurons during training
        """
        super(Decoder, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = FeedForward(d_model, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        """
        Forward pass through the decoder.

        :param src: tensor (batch_size, seq_length, d_model)
        :param tgt: tensor (batch_size, seq_length, d_model)
        :return: tensor (batch_size, seq_length, d_model)
        """
        self_attn_x = self.self_attn(tgt, tgt, tgt)
        x = x + self.dropout(self_attn_x)
        x = self.norm1(x)

        cross_attn_x = self.cross_attn(x, src, src)
        x = x + self.dropout(cross_attn_x)
        x = self.norm2(x)

        ffn_x = self.ffn(x)
        x = x + self.dropout(ffn_x)
        x = self.norm3(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, in_size=3, out_size=5, d_model=512, num_heads=1, num_layers=1, dim_feedforward=2048, dropout=0):
        """
        Initialize the Transformer model.

        :param in_size: size of the input features
        :param out_size: size of the output classes
        :param nhead: number of attention heads
        :param num_layers: number of encoder/decoder layers
        :param dim_feedforward: dimension of the feedforward network hidden layer
        :param dropout: dropout rate
        """
        super(Transformer, self).__init__()

        self.enc_embedding = nn.Embedding(in_size, d_model)
        self.dec_embedding = nn.Embedding(out_size, d_model)

        self.encoder = nn.ModuleList([Encoder(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([Decoder(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        
        self.classifier = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights and biases of the classifier linear layer:
        - Set the bias of the classifier linear layer to zero.
        - Initialize the weights with values drawn from a Xavier uniform distribution.
        """
        self.classifier.bias.data.zero_()
        nn.init.xavier_uniform_(self.classifier.weight.data)
        
    def forward(self, src, tgt):
        """
        Forward pass of the transformer model.

        :param src: tensor (batch_size, seq_length, d_model)
        :param tgt: tensor (batch_size, seq_length, d_model)
        :return: tensor (batch_size, seq_length, out_size)
        """
        src = self.enc_embedding(src)
        src = self.dropout(src)
                                             
        for enc_layer in self.encoder:
            src = enc_layer(src=src)

        tgt = self.dec_embedding(tgt)
        tgt = self.dropout(tgt)

        for dec_layer in self.decoder:
            tgt = dec_layer(src, tgt)

        output = self.classifier(tgt)

        return output
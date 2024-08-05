import torch.nn as nn

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
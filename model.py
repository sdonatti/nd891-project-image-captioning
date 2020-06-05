import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------------------------------------------------
class EncoderCNN(nn.Module):
    """
    A convolutional neural network based on ResNeXt-50-32x4d PyTorch model
    References: `Xie et al. 2015 <http://arxiv.org/abs/1611.05431>`_
    """

    def __init__(self, embed_size, dropout=0.25):
        """
        Defines the neural network layers and initializes trainable parameters

        Parameters
        ----------
        embed_size : int
            The dimensionality of image and word embeddings
        dropout : float
            The dropout probability to use before the fully connected layer
        """
        # initialize parent class variables
        super(EncoderCNN, self).__init__()
        # load pre-trained model and freeze its layers
        resnext = models.resnext50_32x4d(pretrained=True)
        for param in resnext.parameters():
            param.requires_grad_(False)
        # remove the classifier from the pre-trained model
        modules = list(resnext.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        # define dropout layer
        self.dropout = nn.Dropout(dropout)
        # define the embedding layer
        self.embedding = nn.Linear(resnext.fc.in_features, embed_size, bias=False)
        # define the normalization layer
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """
        Generates embedded representations of images

        Parameters
        ----------
        images : torch.Tensor
            The processed images: (N, C, H, W)

        Returns
        -------
        torch.Tensor
            The images embedded representation: (N, embedding size)
        """
        # extract image features
        features = self.extractor(images)
        # add a bit of regularization to prevent overfitting
        features = self.dropout(features.view(features.size(0), -1))
        # embed the image features
        features = self.embedding(features)
        # normalize the image features to speed up learning
        features = self.bn(features)
        return features
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
class DecoderRNN(nn.Module):
    """
    A recurrent neural network based on the Gated Recurrent Unit from PyTorch
    References: `Chung et al. 2014 <http://arxiv.org/abs/1412.3555>`_
    """

    def __init__(self, embed_size, hidden_size, vocab_size, n_layers=1, dropout=0.25):
        """
        Defines the neural network layers and initializes trainable parameters

        Parameters
        ----------
        embed_size : int
            The dimensionality of image and word embeddings
        hidden_size : int
            The size of the hidden layer outputs
        vocab_size : int
            The number of input/output dimensions of the neural network (the size of the vocabulary)
       n_layers : int
            The number of recurrent layers of the neural network
        dropout : float
            The dropout probability in between the recurrent and fully connected layers of the neural network
        """
        # initialize parent class variables
        super(DecoderRNN, self).__init__()
        # set class variables
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        # define embedding and recurrent neural network layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        # define dropout layer
        self.dropout = nn.Dropout(dropout)
        # define linear layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Calculates the vocabulary scores for image features and their captions

        Parameters
        ----------
        features : torch.Tensor
            The images embedded representation: (N, embedding size)
        captions : torch.Tensor
            The processed captions: (N, number of words)

        Returns
        -------
        torch.Tensor
            The vocabulary scores: (N, number of words, vocabulary size)
        """
        # there is no need to predict what comes after end token
        captions = captions[:, :-1]
        # get the embedded representation of the captions
        embeddings = self.embedding(captions)
        # concatenate the image features and the embedded representation of the captions
        rnn_input = torch.cat((features.unsqueeze(dim=1), embeddings), dim=1)
        # feed the image features and the embedded representation of the captions to the rnn layer
        rnn_output, _ = self.rnn(rnn_input)
        # add a bit of regularization to prevent overfitting
        fc_input = self.dropout(rnn_output)
        # feed the rnn layer output to the fully connected layer to calculate the vocabulary scores
        fc_output = self.fc(fc_input)
        # return the vocabulary scores
        return fc_output

    def sample(self, inputs, states=None, max_len=20):
        """
        Generates a caption to describe the content of a given image

        Parameters
        ----------
        inputs : torch.Tensor
            The image embedded representation: (1, embedding size)
        states : torch.Tensor
            The initial hidden state for the recurrent layer of the neural network
        max_len : int
            The maximum number of words for the image caption

        Returns
        -------
        list
            The word indices that represent the image caption
        """
        # generate the predicted words
        word_idx = []
        for _ in torch.arange(max_len):
            # forward pass the inputs through the neural network
            rnn_output, states = self.rnn(inputs, states)
            fc_output = self.fc(rnn_output)
            # get the predicted word with the highest probability
            _, prob_word_idx = fc_output.max(dim=2)
            # add the predicted word to the sentence
            word_idx.append(prob_word_idx.item())
            # check if the predicted word is end token and stop
            if prob_word_idx == 1:
                break
            # calculate the next inputs for the neural network
            inputs = self.embedding(prob_word_idx)
        # return the predicted sentence
        return word_idx
# ----------------------------------------------------------------------------

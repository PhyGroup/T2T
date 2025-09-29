import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        decoder_hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn_fc(torch.cat((decoder_hidden_repeated, encoder_outputs), dim=2)))
        attention_scores = self.v_fc(energy).squeeze(2)
        return torch.softmax(attention_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, attention, bert_embedding_matrix=None,
                 action_embedding_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding.from_pretrained(bert_embedding_matrix,
                                                     freeze=False) if bert_embedding_matrix is not None else nn.Embedding(
            output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim + action_embedding_dim, hidden_dim, num_layers,
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input_token, decoder_hidden, decoder_cell, encoder_outputs, action_embedding):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(decoder_hidden[-1], encoder_outputs)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context_vector, action_embedding.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (decoder_hidden, decoder_cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell, attn_weights.squeeze(1)
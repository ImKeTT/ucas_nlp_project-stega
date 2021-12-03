import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class TextVAE(nn.Module):
	def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, latent_dim, dropout_rate):
		super(TextVAE, self).__init__()
		self._cell = cell
		self._vocab_size = vocab_size
		self._hidden_dim = hidden_dim
		self._num_layers = num_layers
		self._latent_dim = latent_dim

		self.embedding = nn.Embedding(vocab_size, embed_size)
		if cell == 'rnn':
			self.encoder_rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
			self.decoder_rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
		elif cell == 'gru':
			self.encoder_rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
			self.decoder_rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
		elif cell == 'lstm':
			self.encoder_rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
			self.decoder_rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
		else:
			raise Exception('no such rnn cell')

		self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
		self.hidden2logv = nn.Linear(hidden_dim, latent_dim)
		self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
		self.output_layer = nn.Linear(hidden_dim, vocab_size)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x, length):
		x = x.long()
		batch_size = x.size(0)
		sorted_lengths, sorted_idx = torch.sort(length, descending=True)
		x = x[sorted_idx]

		# encoder =============================================================
		x_embeded = self.embedding(x)
		x_packed = rnn_utils.pack_padded_sequence(x_embeded, sorted_lengths.data.tolist(), batch_first=True)
		_, h_last = self.encoder_rnn(x_packed)
		if self._cell == 'lstm':
			h_last = h_last[0]
		h_last = h_last[-1, :, :]

		# reparameterization ==================================================
		mean = self.hidden2mean(h_last)
		logv = self.hidden2logv(h_last)
		std = torch.exp(0.5 * logv)
		z = torch.randn([batch_size, self._latent_dim])
		if torch.cuda.is_available():
			z = z.cuda()
		z = z * std + mean

		# decoder =============================================================
		h_init = torch.zeros([self._num_layers, batch_size, self._hidden_dim])
		if torch.cuda.is_available():
			h_init = h_init.cuda()
		h_init[0, :, :] = self.latent2hidden(z)
		if self._cell == 'lstm':
			c_init = torch.zeros((self._num_layers, batch_size, self._hidden_dim))
			if torch.cuda.is_available():
				c_init = c_init.cuda()
			h_init = (h_init, c_init)
		outputs, _ = self.decoder_rnn(x_packed, h_init)

		total_length = x_embeded.size(1)
		outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)[0]
		outputs = outputs.contiguous()
		_, reversed_idx = torch.sort(sorted_idx)
		outputs = outputs[reversed_idx]
		b, s, _ = outputs.size()

		logits = self.output_layer(outputs.view(b*s, self._hidden_dim))
		log_prob = self.log_softmax(logits)
		log_prob = log_prob.view(b, s, self._vocab_size)

		return log_prob, mean, logv

	def sample(self, x, z, return_logprob=False):
		x = x.long()
		batch_size = x.size(0)
		x_embeded = self.embedding(x)
		h_init = torch.zeros((self._num_layers, batch_size, self._hidden_dim))
		if torch.cuda.is_available():
			h_init = h_init.cuda()
		h_init[0, :, :] = self.latent2hidden(z)
		if self._cell == 'lstm':
			c_init = torch.zeros((self._num_layers, batch_size, self._hidden_dim))
			if torch.cuda.is_available():
				c_init = c_init.cuda()
			h_init = (h_init, c_init)
		outputs, _ = self.decoder_rnn(x_embeded, h_init)
		outputs = outputs.contiguous()
		b, s, _ = outputs.size()
		logits = self.output_layer(outputs.view(b * s, self._hidden_dim))
		log_prob = self.log_softmax(logits)
		log_prob = log_prob.view(b, s, self._vocab_size)

		if return_logprob:
			return log_prob

		prob = torch.exp(log_prob)[:, -1, :]
		prob[:, 1] = 0
		prob = prob / prob.sum()
		return torch.multinomial(prob, 1)

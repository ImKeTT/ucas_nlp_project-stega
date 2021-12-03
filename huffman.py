import numpy as np 
import math
import torch
from utils import *
import Huffman_Encoding

def encode_huffman(bit_stream, model, bpw, vocabulary, config, model_name, device='cpu'):
	bit_index = 0
	bit_stream = ''.join(bit_stream)
	model.eval()
	STRATEGY = config.STRATEGY
	with torch.no_grad():
		for bit in [bpw]:
			stega_text = []
			stega_bits = []
			while bit_index < len(bit_stream):
				stega_sentence = []
				stega_bit = ''
				if model_name == "VAE":
					z = torch.randn([config.LATENT_DIM]).to(device)
					x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
					samp = model.sample(x, z)
				elif model_name == "RNN":
					x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
					samp = model.sample(x)
				stega_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
				x = torch.cat([x, samp], dim=1)

				for i in range(config.MAX_GENERATE_LENGTH - 1):
					if '_EOS' in stega_sentence:
						break
					# conditional probability distribution
					log_prob = model.sample(x, z, return_logprob=True) if model_name=="VAE" else model(x)
					prob = torch.exp(log_prob)[:, -1, :].reshape(-1)
					prob[1] = 0
					prob = prob / prob.sum()
					if STRATEGY == 'topk':
						prob, indices = prob.sort(descending=True)
						prob = prob[:2**bit]
						indices = indices[:2**bit]
					elif STRATEGY == 'threshold':
						prob, indices = prob.sort(descending=True)
						max_index = (prob > prob[0]/(2**bit)).nonzero()
						max_index = int(max_index[-1])
						if max_index == 0:
							gen = int(indices[0])
							stega_sentence += [vocabulary.i2w[gen]]
							x = torch.cat([x, torch.LongTensor([[gen]]).to(device)], dim=1).to(device)
							continue
						prob = prob[:max_index+1]
						indices = indices[:max_index+1]
					elif STRATEGY == 'sample':
						indices = torch.multinomial(prob, 2**bit)
						prob = prob[indices]
					else:
						raise Exception(f'no {STRATEGY} strategy !')

					# huffman coding
					nodes = Huffman_Encoding.createNodes([_ for _ in prob])
					root = Huffman_Encoding.createHuffmanTree(nodes)
					codes = Huffman_Encoding.huffmanEncoding(nodes, root)
					# choose word
					for i in range(2**bit):
						if bit_stream[bit_index:bit_index + i + 1] in codes:
							code_index = codes.index(bit_stream[bit_index:bit_index + i + 1])
							gen = int(indices[code_index])
							stega_sentence += [vocabulary.i2w[gen]]
							if vocabulary.i2w[gen] == '_EOS':
								break
							x = torch.cat([x, torch.LongTensor([[gen]]).to(device)], dim=1).to(device)
							stega_bit += bit_stream[bit_index:bit_index + i + 1]
							bit_index = bit_index + i + 1
							break
				# check
				if '_EOS' in stega_sentence:
					stega_sentence.remove('_EOS')
				if (len(stega_sentence) <= config.MAX_LEN) and (len(stega_sentence) >= config.MIN_LEN):
					stega_text.append(stega_sentence)
					stega_bits.append(stega_bit)
	return stega_text, stega_bits
	"""
			# write files
			with open('stego/' + DATASET + '/huffman-' + STRATEGY + '-' + str(bit) + 'bit.txt', 'w', encoding='utf8') as f:
				for sentence in stega_text:
					f.write(' '.join(sentence) + '\n')
			with open('stego/' + DATASET + '/huffman-' + STRATEGY + '-' + str(bit) + 'bit.bit', 'w', encoding='utf8') as f:
				for bits in stega_bits:
					f.write(bits + '\n')
	"""
def decode_huffman(stega_text, model, bit, vocabulary, config, model_name, device='cpu'):
	model.eval()
	STRATEGY = config.STRATEGY
	with torch.no_grad():
		# with open('stego/' + DATASET + '/huffman-' + STRATEGY + '-' + str(bit) + 'bit.txt', 'r', encoding='utf8') as f:
		# 	stega_text = f.readlines()
		# with open('stego/' + DATASET + '/huffman-' + STRATEGY + '-' + str(bit) + 'bit.bit', 'r', encoding='utf8') as f:
		# 	stega_bits = f.readlines()
		decode_bit = ''
		for index in range(len(stega_text)):
			stega_sentence = stega_text[index].strip()
			start_word = stega_sentence.split()[0]
			x = torch.LongTensor([[vocabulary.w2i['_BOS'], vocabulary.w2i[start_word]]]).to(device)
			if model_name == "VAE":
				z = torch.randn([config.LATENT_DIM]).to(device)

			for word in stega_sentence.split()[1:]:
				# conditional probability distribution
				log_prob = model.sample(x, z, return_logprob=True) if model_name=="VAE" else model(x)
				prob = torch.exp(log_prob)[:, -1, :].reshape(-1)
				prob[1] = 0                             # set unk to zero
				prob = prob / prob.sum()
				if STRATEGY == 'topk':
					prob, indices = prob.sort(descending=True)
					prob = prob[:2**bit]
					indices = indices[:2**bit]
				elif STRATEGY == 'threshold':
					prob, indices = prob.sort(descending=True)
					max_index = (prob > TH[bit - 1]).nonzero()
					if len(max_index) < 2:
						x = torch.cat([x, torch.LongTensor([[vocabulary.w2i[word]]]).to(device)], dim=1).to(device)
						continue
					max_index = int(max_index[-1])
					prob = prob[:max_index+1]
					indices = indices[:max_index+1]
				else:
					raise Exception('no such strategy')

				# huffman coding
				nodes = Huffman_Encoding.createNodes([_ for _ in prob])
				root = Huffman_Encoding.createHuffmanTree(nodes)
				codes = Huffman_Encoding.huffmanEncoding(nodes, root)
				# secret message
				decode_bit += codes[(indices == vocabulary.w2i[word]).nonzero()]

				x = torch.cat([x, torch.LongTensor([[vocabulary.w2i[word]]]).to(device)], dim=1).to(device)
	return decode_bit
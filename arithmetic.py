import numpy as np 
import math
import torch
from utils import *
import random
from gpt_2.utils import *


def encode_arithmetic(bit_stream, model, bpw, vocabulary, config, model_name, seed, device='cpu'):
	torch.manual_seed(seed)
	random.seed(seed)
	bit_index = 0
	bit_stream = ''.join(bit_stream)

	model.eval()
	STRATEGY = config.STRATEGY
	with torch.no_grad():
		for bit in [bpw]:
			stega_text = []
			stega_bits = []
			while bit_index<len(bit_stream):
				# while len(stega_text) < config.GENERATE_NUM:
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

				max_val = 2 ** config.PRECISION  # num of intervals
				cur_interval = [0, max_val]  # bottom inclusive, top exclusive
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
						prob = prob[:max_index + 1]
						indices = indices[:max_index + 1]
					elif STRATEGY == 'sample':
						indices = torch.multinomial(prob, 2 ** bit)
						prob = prob[indices]
					elif STRATEGY == '':
						prob, indices = prob.sort(descending=True)
					else:
						raise Exception(f'no {STRATEGY} strategy !')

					# arithmetic coding
					cur_int_range = cur_interval[1] - cur_interval[0]   # 区间的大小  2^26
					cur_threshold = 1 / cur_int_range                   # 每个区间多大
					if prob[-1] < cur_threshold:
						k = max(2, (prob < cur_threshold).nonzero()[0].item())
						prob = prob[:k]
						indices = indices[:k]

					prob = prob / prob.sum()                        # 截断后线性归一化
					prob = prob.double()
					prob *= cur_int_range                           # 概率转换为多少个区间
					prob = prob.round().long()                      # 四舍五入取整，区间数描述的概率

					cum_probs = prob.cumsum(0)		# 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
					overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])
					if len(overfill_index) > 0:
						cum_probs = cum_probs[:overfill_index[0]]           # [299] 去掉最后一个概率
					cum_probs += cur_int_range - cum_probs[-1]              # 分布函数加到和区间数相等，区间数表示的分布函数

					cum_probs += cur_interval[0]                            # 分布函数的第一项从左区间开始

					message_bits = bit_stream[bit_index: bit_index + config.PRECISION]     # 取了26位，但不是编码这26位，是用这26位锁定一个位置
					message_bits = [int(_) for _ in message_bits]
					message_idx = bits2int(reversed(message_bits))                  # reverse只是为了计算int
					selection = (cum_probs > message_idx).nonzero()[0].item()       # 选择的单词的索引，int，选择第几个单词

					new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
					new_int_top = cum_probs[selection]

					new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, config.PRECISION)))  # 二进制的下边界
					new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, config.PRECISION)))  # 二进制的上边界

					num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

					new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
					new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

					cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
					cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

					prev = indices[selection].view(1, 1)  # 一个数，代表选了哪个单词
					stega_sentence.append(vocabulary.i2w[int(prev)])
					if vocabulary.i2w[int(prev)] == '_EOS':
						break
					x = torch.cat([x, prev], dim=1)
					stega_bit += bit_stream[bit_index:bit_index + num_bits_encoded]
					bit_index += num_bits_encoded

				# check
				if '_EOS' in stega_sentence:
					stega_sentence.remove('_EOS')
					# bit_index -= num_bits_encoded
				if (len(stega_sentence) <= config.MAX_LEN) and (len(stega_sentence) >= config.MIN_LEN):
					stega_text.append(stega_sentence)
					stega_bits.append(stega_bit)
	return stega_text, stega_bits
"""
			# write files
			with open('stego/' + DATASET + '/arithmetic-' + STRATEGY + '-' + str(bit) + 'bit.txt', 'w', encoding='utf8') as f:
				for sentence in stega_text:
					f.write(' '.join(sentence) + '\n')
			with open('stego/' + DATASET + '/arithmetic-' + STRATEGY + '-' + str(bit) + 'bit.bit', 'w', encoding='utf8') as f:
				for bits in stega_bits:
					f.write(bits + '\n')
"""


def encode_arithmetic_gpt2(bit_stream, model, bpw, enc, config, device='cpu'):
	bit_index = 0
	bit_stream = ''.join(bit_stream)

	model.eval()
	STRATEGY = config.STRATEGY
	with torch.no_grad():
		for bit in [bpw]:
			stega_text = []
			stega_bits = []
			# while bit_index<len(bit_stream):
			while bit_index < len(bit_stream):
				stega_sentence = []
				stega_bit = ''
				context = \
				"""Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.
				"""
				context_tokens = encode_context(context, enc)
				context = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
				prev = context
				output = context
				past = None
				max_val = 2 ** config.PRECISION  # num of intervals
				cur_interval = [0, max_val]  # bottom inclusive, top exclusive
				for i in range(config.MAX_GENERATE_LENGTH - 1):
					if '_EOS' in stega_sentence:
						break
					# conditional probability distribution
					log_prob, past = model(prev.unsqueeze(0), past=past)
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
						prob = prob[:max_index + 1]
						indices = indices[:max_index + 1]
					elif STRATEGY == 'sample':
						indices = torch.multinomial(prob, 2 ** bit)
						prob = prob[indices]
					elif STRATEGY == '':
						prob, indices = prob.sort(descending=True)
					else:
						raise Exception('no such strategy')

					# arithmetic coding
					cur_int_range = cur_interval[1] - cur_interval[0]   # 区间的大小  2^26
					cur_threshold = 1 / cur_int_range                   # 每个区间多大
					if prob[-1] < cur_threshold:
						k = max(2, (prob < cur_threshold).nonzero()[0].item())
						prob = prob[:k]
						indices = indices[:k]

					prob = prob / prob.sum()                        # 截断后线性归一化
					prob = prob.double()
					prob *= cur_int_range                           # 概率转换为多少个区间
					prob = prob.round().long()                      # 四舍五入取整，区间数描述的概率

					cum_probs = prob.cumsum(0)		# 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
					overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])
					if len(overfill_index) > 0:
						cum_probs = cum_probs[:overfill_index[0]]           # [299] 去掉最后一个概率
					cum_probs += cur_int_range - cum_probs[-1]              # 分布函数加到和区间数相等，区间数表示的分布函数

					cum_probs += cur_interval[0]                            # 分布函数的第一项从左区间开始

					message_bits = bit_stream[bit_index: bit_index + config.PRECISION]     # 取了26位，但不是编码这26位，是用这26位锁定一个位置
					message_bits = [int(_) for _ in message_bits]
					message_idx = bits2int(reversed(message_bits))                  # reverse只是为了计算int
					selection = (cum_probs > message_idx).nonzero()[0].item()       # 选择的单词的索引，int，选择第几个单词

					new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
					new_int_top = cum_probs[selection]

					new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, config.PRECISION)))  # 二进制的下边界
					new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, config.PRECISION)))  # 二进制的上边界

					num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

					new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
					new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

					cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
					cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

					prev = indices[selection].view(1, 1)  # 一个数，代表选了哪个单词
					stega_sentence.append(enc.decode(int(prev)))
					if enc.decode(int(prev)) == '_EOS':
						break
					output = torch.cat([output, prev], dim=1)
					stega_bit += bit_stream[bit_index:bit_index + num_bits_encoded]
					bit_index += num_bits_encoded

				# check
				if '_EOS' in stega_sentence:
					stega_sentence.remove('_EOS')
				if (len(stega_sentence) <= config.MAX_LEN) and (len(stega_sentence) >= config.MIN_LEN):
					stega_text.append(stega_sentence)
					stega_bits.append(stega_bit)
	return stega_text, stega_bits


def decode_arithmetic(stega_text, model, bit, vocabulary, config, model_name, seed, device='cpu'):
	torch.manual_seed(seed)
	random.seed(seed)
	model.eval()
	STRATEGY = config.STRATEGY
	with torch.no_grad():
		decode_bit = ''
		# with open('stego/' + DATASET + '/arithmetic-' + STRATEGY + '-' + str(bit) + 'bit.txt', 'r', encoding='utf8') as f:
		# 	stega_text = f.readlines()
		# with open('stego/' + DATASET + '/arithmetic-' + STRATEGY + '-' + str(bit) + 'bit.bit', 'r', encoding='utf8') as f:
		# 	stega_bits = f.readlines()
		for index in range(len(stega_text)):
			stega_sentence = stega_text[index].strip()
			start_word = stega_sentence.split()[0]
			x = torch.LongTensor([[vocabulary.w2i['_BOS'], vocabulary.w2i[start_word]]]).to(device)
			if model_name == "VAE":
				z = torch.randn([config.LATENT_DIM]).to(device)
			max_val = 2 ** config.PRECISION  # num of intervals
			cur_interval = [0, max_val]  # bottom inclusive, top exclusive
			for word in stega_sentence.split()[1:]:
				# conditional probability distribution
				log_prob = model.sample(x, z, return_logprob=True) if model_name=="VAE" else model(x)
				prob = torch.exp(log_prob)[:, -1, :].reshape(-1)
				prob[1] = 0
				prob = prob / prob.sum()
				if STRATEGY == 'topk':
					prob, indices = prob.sort(descending=True)
					prob = prob[:2**bit]
					indices = indices[:2**bit]
				elif STRATEGY == 'sample':
					indices = torch.multinomial(prob, 2 ** bit)
					prob = prob[indices]
				elif STRATEGY == 'threshold':
					prob, indices = prob.sort(descending=True)
					max_index = (prob > TH[bit - 1]).nonzero()
					if len(max_index) < 2:
						x = torch.cat([x, torch.LongTensor([[vocabulary.w2i[word]]]).to(device)], dim=1).to(device)
						continue
					max_index = int(max_index[-1])
					prob = prob[:max_index + 1]
					indices = indices[:max_index + 1]
				elif STRATEGY == '':
					prob, indices = prob.sort(descending=True)
				else:
					raise Exception(f'NO {STRATEGY} strategy !')

				# arithmetic coding
				cur_int_range = cur_interval[1] - cur_interval[0]   # 区间的大小  2^26
				cur_threshold = 1 / cur_int_range                   # 每个区间多大
				if prob[-1] < cur_threshold:
					k = max(2, (prob < cur_threshold).nonzero()[0].item())
					prob = prob[:k]
					indices = indices[:k]

				prob = prob / prob.sum()                        # 截断后线性归一化
				prob *= cur_int_range                           # 概率转换为多少个区间
				prob = prob.round().long()                     # 四舍五入取整，区间数描述的概率

				cum_probs = prob.cumsum(0)                              # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
				overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])
				if len(overfill_index) > 0:
					cum_probs = cum_probs[:overfill_index[0]]           # [299] 去掉最后一个概率
				cum_probs += cur_int_range - cum_probs[-1]              # 分布函数加到和区间数相等，区间数表示的分布函数

				cum_probs += cur_interval[0]                            # 分布函数的第一项从左区间开始
				selection = (indices == vocabulary.w2i[word]).nonzero().item()

				new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
				new_int_top = cum_probs[selection]

				new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, config.PRECISION)))  # 二进制的下边界
				new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, config.PRECISION)))  # 二进制的上边界

				num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

				# start decoding
				new_bits = new_int_top_bits_inc[:num_bits_encoded]
				decode_bit += ''.join([str(i) for i in new_bits])

				new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
				new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

				cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
				cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

				x = torch.cat([x, torch.LongTensor([[vocabulary.w2i[word]]]).to(device)], dim=1).to(device)
	return decode_bit
	"""
			if decode_bit != stega_bit:
				print(stega_bit)
				print(stega_sentence)
				print(decode_bit)

	print('succeed')
	"""


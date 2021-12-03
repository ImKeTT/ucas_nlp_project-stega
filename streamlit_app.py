import streamlit as st
import numpy as np 
# from gpt_2.utils import get_model, encode_context
from rnn import lm
from vae import textvae
import torch
import random
from rnn import lm
import arithmetic as ac 
import huffman as huf
import os
import gpt_2.arithmetic as gpt_ac
import gpt_2.huffman_baseline as gpt_huf

from gpt_2 import utils as gpt_utils

import bitarray
from utils import *

class Config:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def main(secret_message="test", model_name="VAE", mode="arithmetic"):
    ## Trun words to bit message
    bpw = 999 if model_name=="arithmetic" else 3
    seed = 415
    message_str = secret_message
    ba = bitarray.bitarray()
    ba.frombytes(message_str.encode('utf-8'))
    message = ba.tolist()
    message = [str(int(item)) for item in  message]
    config_dict = load_json(f"{model_name.lower()}/config.json")[0]
    config = Config(**config_dict)
    random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'
    if model_name == "RNN" or model_name == "VAE":
        data_path = '_data/' + config.DATASET + '2020.txt'
        vocabulary = Vocabulary(
			data_path,
			max_len=config.MAX_LEN,
			min_len=config.MIN_LEN,
			word_drop=config.WORD_DROP
		)
        model = lm.LM(
			cell=config.CELL,
			vocab_size=vocabulary.vocab_size,
			embed_size=config.EMBED_SIZE,
			hidden_dim=config.HIDDEN_DIM,
			num_layers=config.NUM_LAYERS,
			dropout_rate=config.DROPOUT_RATE
		) if model_name == "RNN" else textvae.TextVAE(
			cell=config.CELL,
			vocab_size=vocabulary.vocab_size,
			embed_size=config.EMBED_SIZE,
			hidden_dim=config.HIDDEN_DIM,
			num_layers=config.NUM_LAYERS,
			latent_dim=config.LATENT_DIM,
			dropout_rate=config.DROPOUT_RATE
		)
        model.to(device)
        # total_params = sum(p.numel() for p in model.parameters())
        # print("Total params: {:d}".format(total_params))
        # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("Trainable params: {:d}".format(total_trainable_params))
        model.load_state_dict(torch.load(f'{model_name.lower()}/models/{config.DATASET}.pkl', map_location=device))
        # print('checkpoint loaded')
        # print()
        if mode == "arithmetic":
            stega_text, stega_bits = ac.encode_arithmetic(message, model, bpw, vocabulary, config, model_name, seed, device=device)
            stego = [' '.join(stext) for stext in stega_text]
            stego = '. '.join(stego)
            # print("stego:", '. '.join(stego))
            # try:
            #     message_rec = ac.decode_arithmetic(stego, model, bpw, vocabulary, config, model_name, seed, device=device)
            # except:
            #     message_rec = message
            # message_rec = ' '.join(message_rec).split()
            # message_rec = [int(item) for item in message_rec]
            # ba = bitarray.bitarray(message_rec)
            # reconst = ba.tobytes().decode('utf-8', 'ignore')
            # print("reconst:", reconst)
        elif mode == "huffman":
            stega_text, stega_bits = huf.encode_huffman(message, model, bpw, vocabulary, config, model_name, device=device)
            stego = [' '.join(stext) for stext in stega_text]
            stego = '. '.join(stego)
            # print("stego:", '. '.join(stego))
            # try:
            #     message_rec = huf.decode_huffman(stego, model, bpw, vocabulary, config, model_name, device=device)
            # except:
            #     message_rec = message
            # message_rec = ' '.join(message_rec).split()
            # message_rec = [int(item) for item in message_rec]
            # ba = bitarray.bitarray(message_rec)
            # reconst = ba.tobytes().decode('utf-8', 'ignore')
            # print("reconst:", reconst)
        else:
            raise Exception(f'NO {mode} steganograpy method !')
    elif model_name == "GPT_2":
        message = ba.tolist()
        enc, model = gpt_utils.get_model(model_name='gpt2')
        context = \
            """Washington received his initial military training and command 
            with the Virginia Regiment during the French and Indian War. He was 
            later elected to the Virginia House of Burgesses and was named a delegate 
            to the Continental Congress, where he was appointed Commanding General of the nation's 
            Continental Army. Washington led American forces, allied with France, in the defeat of 
            the British at Yorktown. Once victory for the United States was in hand in 1783, 
            Washington resigned his commission.
            """
        context_tokens = gpt_utils.encode_context(context, enc)
        if mode == 'arithmetic':
            out, nll, kl, words_per_bit, Hq = gpt_ac.encode_arithmetic(model, enc, message, context_tokens, temp=config.temp,
                                                                finish_sent=False, precision=config.PRECISION, topk=config.topk, device=device)
        elif mode == 'huffman':
            out, nll, kl, words_per_bit = gpt_huf.encode_huffman(model, enc, message, context_tokens, config.block_size,
                                                         finish_sent=False, device=device)
        stego = enc.decode(out)
        # if mode == 'arithmetic':
        #     message_rec = gpt_ac.decode_arithmetic(model, enc, stego, context_tokens, temp=config.temp, precision=config.PRECISION, topk=config.topk, device=device)
        # elif mode == 'huffman':
        #     message_rec = gpt_huf.decode_huffman(model, enc, stego, context_tokens, config.block_size, device=device)
        # message_rec = [bool(item) for item in message_rec]
        # ba = bitarray.bitarray(message_rec)
        # reconst = ba.tobytes().decode('utf-8', 'ignore')
        # print("reconst:",reconst)
    else:
        raise Exception(f'NO {model_name} language model for steganography !')
    return stego

def extract(stego, model_name="VAE", mode="arithmetic"):
    ## Trun words to bit message
    bpw = 999 if model_name == "arithmetic" else 3
    seed = 415
    # print(message)
    config_dict = load_json(f"{model_name.lower()}/config.json")[0]
    config = Config(**config_dict)
    random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'
    if model_name == "RNN" or model_name == "VAE":
        stego = stego.split('. ')
        data_path = '_data/' + config.DATASET + '2020.txt'
        vocabulary = Vocabulary(
            data_path,
            max_len=config.MAX_LEN,
            min_len=config.MIN_LEN,
            word_drop=config.WORD_DROP
        )
        model = lm.LM(
            cell=config.CELL,
            vocab_size=vocabulary.vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE
        ) if model_name == "RNN" else textvae.TextVAE(
            cell=config.CELL,
            vocab_size=vocabulary.vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            latent_dim=config.LATENT_DIM,
            dropout_rate=config.DROPOUT_RATE
        )
        model.to(device)
        # total_params = sum(p.numel() for p in model.parameters())
        # print("Total params: {:d}".format(total_params))
        # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("Trainable params: {:d}".format(total_trainable_params))
        model.load_state_dict(torch.load(f'{model_name.lower()}/models/{config.DATASET}.pkl', map_location=device))
        # print()
        if mode == "arithmetic":
            message_rec = ac.decode_arithmetic(stego, model, bpw, vocabulary, config, model_name, seed, device=device)
            message_rec = ' '.join(message_rec).split()
            message_rec = [int(item) for item in message_rec]
            ba = bitarray.bitarray(message_rec)
            reconst = ba.tobytes().decode('utf-8', 'ignore')
            print("reconst:", reconst)
        elif mode == "huffman":
            message_rec = huf.decode_huffman(stego, model, bpw, vocabulary, config, model_name, device=device)
            message_rec = ' '.join(message_rec).split()
            message_rec = [int(item) for item in message_rec]
            ba = bitarray.bitarray(message_rec)
            reconst = ba.tobytes().decode('utf-8', 'ignore')
        else:
            raise Exception(f'NO {mode} steganograpy method !')
    elif model_name == "GPT_2":
        enc, model = gpt_utils.get_model(model_name='gpt2')
        context = \
            """Washington received his initial military training and command 
            with the Virginia Regiment during the French and Indian War. He was 
            later elected to the Virginia House of Burgesses and was named a delegate 
            to the Continental Congress, where he was appointed Commanding General of the nation's 
            Continental Army. Washington led American forces, allied with France, in the defeat of 
            the British at Yorktown. Once victory for the United States was in hand in 1783, 
            Washington resigned his commission.
            """
        context_tokens = gpt_utils.encode_context(context, enc)
        if mode == 'arithmetic':
            message_rec = gpt_ac.decode_arithmetic(model, enc, stego, context_tokens, temp=config.temp,
                                                   precision=config.PRECISION, topk=config.topk, device=device)
        elif mode == 'huffman':
            message_rec = gpt_huf.decode_huffman(model, enc, stego, context_tokens, config.block_size, device=device)
        message_rec = [bool(item) for item in message_rec]
        ba = bitarray.bitarray(message_rec)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
    else:
        raise Exception(f'NO {model_name} language model for steganography !')
    return reconst

if __name__=="__main__":
    # desc = "Uses RNN/VAE/GPT-2 with Huffman and Arithmetic Coding to hide or extract secret messages. Check out the code and corresponding papers [here](https://github.com/ImKeTT/ucas_nlp_project_stega)!"
    # st.title('Linguistic Generative Steganography')
    # st.write(desc)
    # model_name = st.selectbox('Select a Language Model for Hiding', ('GPT_2', 'RNN', 'VAE'))
    # method = st.selectbox('Select a Coding Method for Hiding', ('huffman', 'arithmetic'))
    # secret_message = st.text_input('Seceret Message (cannot leave blank)')
    # if st.button('Hide Message'):
    #     stego = main(secret_message, model_name, method)
    #     st.write(stego)
    # model_name = st.selectbox('Select a Language Model for Extracting', ('GPT_2', 'RNN', 'VAE'))
    # method = st.selectbox('Select a Coding Method for Extracting', ('huffman', 'arithmetic'))
    # stego = st.text_input('Message to be Extracted (cannot leave blank)')
    # if st.button('Extract Message'):
    #     stego = main(secret_message, model_name, method)
    #     st.write(stego)
    stego = main("this is a secret", "RNN", "huffman")
    print(stego)

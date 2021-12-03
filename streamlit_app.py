#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: streamlit_app.py
@author: ImKe at 2021/12/3
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
from main_func import main, extract
import streamlit as st

desc = "Uses RNN/VAE/GPT-2 with Huffman and Arithmetic Coding to hide or extract secret messages. Check out the code and corresponding papers [here](https://github.com/ImKeTT/ucas_nlp_project-stega)!"
st.title('Linguistic Generative Steganography')
st.write(desc)
model_name = st.selectbox('Select a Language Model for Hiding', ('GPT_2', 'RNN', 'VAE'))
method = st.selectbox('Select a Coding Method for Hiding', ('huffman', 'arithmetic'))
secret_message = st.text_input('Seceret Message (cannot leave blank)')
if st.button('Hide Message'):
    stego = main(secret_message, model_name, method)
    st.write(stego)
model_name = st.selectbox('Select a Language Model for Extracting', ('GPT_2', 'RNN', 'VAE'))
method = st.selectbox('Select a Coding Method for Extracting', ('huffman', 'arithmetic'))
stego = st.text_input('Message to be Extracted (cannot leave blank)')
if st.button('Extract Message'):
    stego = extract(stego, model_name, method)
    st.write(stego)
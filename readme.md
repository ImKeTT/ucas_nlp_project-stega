# UCAS NLP Project Stega: Generative Liguisitic Steganography

首先创建环境，从终端进入在本目录，执行

```bash
pip install -r requirements.txt
```

RNN和VAE模型的权重文件在[百度网盘](https://pan.baidu.com/s/1eQ9WswRhxvuOp_0BJ9T_Rw)，密码`4avb`，分别下载文件夹`vae`和`rnn`中的`models`文件夹，分别放到作业对应的`vae`和`rnn`文件夹下面

接着在终端运行

```bash
python app.py
```

在默认游览器的` localhost:8080`可以使用网页。

文件分布

```bash
.
|____arithmetic.py
|____requirements.txt
|____main_func.py
|______init__.py
|____huffman.py
|____readme.md
|____utils.py
|____app.py
|____templates
| |____index.html
| |____extract.html
| |____static
| |____hide.html
|_____data
| |____movie2020.txt_vocab.txt
| |____tweet2020.txt_vocab.txt
| |____news2020.txt_vocab.txt
|____Huffman_Encoding.py
|____rnn
| |____config.json
| |____models
| | |____tweet.pkl
| |____utils.py
| |____lm.py
|____vae
| |____config.json
| |____textvae.py
| |____models
| | |____tweet.pkl
| |____utils.py
|____gpt_2
| |____arithmetic.py
| |____config.json
| |____huffman.py
| |____utils.py
| |____run_single.py
| |____sample.py
| |____huffman_baseline.py
```


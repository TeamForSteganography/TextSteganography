# Overview

由第十五组完成的文本隐写项目--“情书生成器”

项目包含前端静态界面以及关键代码run_single.py。

情书生成器采用四种生成式文本隐写算法，采用GPT-2模型完成文本生成。

在Harvard NLP的项目即 Ziegler的项目基础之上增加自适应算术编码功能，有效提升约13%的不可感知度以及隐写效率。



# 参考论文(Reference)

《Tina Fang, Martin Jaggi, and Katerina J. Argyraki. 2017. Generating steganographic text with lstms. In *ACL*.》

《Zachary M. Ziegler, Yuntian Deng, and Alexander M. Rush. 2019. Neural linguistic steganography. In *EMNLP*.》

《Falcon Z. Dai and Zheng Cai. 2019. Towards near- imperceptible steganographic text. 》

《Zhong-Liang Yang, Xiaoqing Guo, Zi-Ming Chen, Yongfeng Huang, and Yu-Jin Zhang. 2019b. Rnn- stega: Linguistic steganography based on recurrent neural networks.》



# 参考项目

- https://github.com/YangzlTHU
- https://github.com/RemiRosenthal/TextStegano
- https://github.com/fme-mmt/Neural-steganography
- https://github.com/CQuintt/TwitterStego
- https://github.com/rodrigorivera/mds20_stega



# 描述(Description)

```python
# -----------------------------------------------------
# |       Harvard NLP project edited by Kieran        |
# | Featuring SAAC, Arithmetic, Bins, Huffman implem- |
# | entations on linguistic Steganography based on Te-|
# | xt Generation Language Model. The Basic openAI GPT|
# | -2 language model have been included in the direc |
# | tory pretrained_model. Usage mentioned below.     |
# -----------------------------------------------------

# Usage:
# python run_single.py [-mode] [-unicode_enc] [-block_size] [-temp] [-precision] [-topk] [-nucleus] [-device] [-finish_sent] [-delta] [-language_model]

# Simply Usage:
# python run_single.py
# python run_single.py -mode "huffman"
# python run_single.py -mode "saac" -nucleus 0.98

# API likely:
# message_str: string to be hidden. 需要被隐写的人名，比如 'Kieran'
# context: the context related to the text generation procedure. 上下文CONTEXT，此处更改为使用同目录中其他文件
# message: Binary stream Based on message_str. text --arithmetic encode--> binary stream 根据隐写信息（人名）编码得到的二进制流
# text: covertext. generated text that contains secret information. 生成的含有隐写信息的文本 COVERTEXT
# message_rec: binary stream extracted from stego_text. 对隐写文本进行隐写提取得到的二进制流
# reconst: Decoded text. message_rec --arithmetic decode--> reconst  将隐写提取得到的二进制流进行解码得到的结果，合法输入应该也为人名
# covertext_list: 将所有人名变化得到的covertext保存到的一个list中，可供调用。

# env: Windows 10, python 3.6.12, torch 1.0.1, pytorch_transformers 1.1.0,
#      bitarray 1.0.1, CUDA 10, GTX1050.
```



# Requirement

- torch==1.0.1
- pytorch_transformers>=1.1.0
- bitarray
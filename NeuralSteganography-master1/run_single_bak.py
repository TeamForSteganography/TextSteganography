import numpy as np
import bitarray
import sys
import re
import math
import argparse
import csv
from utils import get_model, encode_context, dfs

from arithmetic import encode_arithmetic, decode_arithmetic
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from sample import sample
from saac import encode_saac, decode_saac
# from base64 import *
import re
import pandas as pd

# -----------------------------------------------------
# |       Harvard NLP project edited by Kieran        |
# | Feature SAAC, Arithmetic, Bins, Huffman implemen- |
# | tations on linguistic Steganography based on Text |
# | Generation Language Model. The Basic openAI GPT-2 |
# | language model have been included in the directo- |
# | ry pretrained_model. Usage mentioned below.       |
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


def main(args):
    # Initial process
    args = vars(args)
    unicode_enc = args['unicode_enc']   # 选择编码方式 
    mode = args['mode']                 # 选择隐写算法
    block_size = args['block_size']     # 隐写参数batch_size
    temp = args['temp']                 # 隐写参数TEMPERATURE，注意下文中最好不要新建temp变量
    precision = args['precision']       # 隐写参数
    topk = args['topk']                 # 文本生成相关参数
    device = args['device']             # device，文本生成相关参数，选择GPU/CPU，默认'cuda'
    finish_sent = args['finish_sent']   # 隐写参数
    nucleus = args['nucleus']           # saac相关隐写参数
    delta = args['delta']               # saac相关隐写参数
    model_name = args['language_model'] # 文本生成模型
    context_file = args['context_file'] # 上下文文件的位置
    message_str = args['name']
    # sample_tokens = 100               # 测试用变量

    # PARAMETERS 默认第一次的隐写信息(人名)
    # message_str = "Chhenl"              # string to be hidden.

    # VALIDATE PARAMETERS 验证隐写算法
    if mode not in ['arithmetic', 'huffman', 'bins', 'saac']:
        raise NotImplementedError
    
    # 打印隐写信息(人名)
    print("Default plain_text is ", message_str)
    
    # 读取上下文
    f = open(context_file, 'r', encoding='utf-8')
    context = f.read()
    f.close()
    print("sample context is ", context)   # related to the text generation procedure.

    # 加载文本生成模型
    print("loading GPT-2 LM to GPU")
    enc, model = get_model(model_name=model_name)
    print("finish loading !")

    print("implication of {}".format(mode))
    
    # bins隐写算法的处理
    if mode == 'bins':
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)

    # saac隐写算法的处理
    if delta and mode == "saac":
        nucleus = 2 ** (-1.0 * delta)



    # 以下注释都为旧调试过程中的注释
    # fix situation: directly encode the text.
    # print("directly encode the plain txt:\n", enc.encode(message_str))
    # print("Decode back:\n", enc.decode(enc.encode(message_str)))

    # can ensure the problem arise in the arithmetic_decode as well as the arithmetic_encode function.

    # ----------------------start test----------------------------
    # test_str = "hello world."
    # print("test_str = ", test_str)
    # out = enc.encode(test_str)
    # print("out = ", out)
    # decode_str = enc.decode(out)
    # print("decode_str = ", decode_str)
    # print("enc.encode(decode_str) = ", enc.encode(decode_str))
    # ----------------------stop test-----------------------------

    # Archive Basic Initialization----------------------------------
    # print("plain_text is {}".format(message_str))
    # unicode_enc = False
    # mode = 'huffman'
    # block_size = 3 # for huffman and bins
    # temp = 0.9 # for arithmetic
    # precision = 26 # for arithmetic
    # sample_tokens = 100 # for sample, delete sample
    # topk = 300
    # device = 'cuda'
    # finish_sent=False # whether or not to force finish sent. If so, stats displayed will be for non-finished sentence
    # nucleus = 0.95
    # Archive Basic Initialization----------------------------------






    first_flag = 1  # 对下文中默认处理的标志
    context_tokens = encode_context(context, enc)   # 对context进行语言模型相关的编码

    while(1):
        # ---此处在循环中，则会不断等待输入隐写信息（人名）--------------------------------------
        # ------------------------------------------------------------------------------------
        # list_for_bpw = [] # 用于计算Bits/word参数
        # list_for_DKL = [] # 用于计算KL参数
        # list_for_seq = [] # 用于标记
        
        if first_flag == 0:
            message_str = input("Please reenter a new plaintext:")
            # output_amount = len(message_str)
        
        # 得到对隐写信息（人名）的大小写集合
        message_str = message_str.upper()
        arr=list(message_str)
        generated_array = dfs(arr,0,[])
        
        first_flag = 0
        covertext_list = []
        
        for temp_count in range(0, len(generated_array)):
            # First encode message to uniform bits, without any context
            # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
            
            # if temp_count > 10:
            #     break                 # 测试时最好完成修正，此处限制输出10个COVERTEXT
            
            print("="*80)
            print("Altering the #{} msg_str:".format(temp_count), message_str)
            message_str = generated_array[temp_count]           # 选择一个隐写信息(比如 KiErAn)



            # 得到message。即上文所述的字节流
            if unicode_enc:
                ba = bitarray.bitarray()
                ba.frombytes(message_str.encode('utf-8'))
                message = ba.tolist()
            else:
                message_ctx = [enc.encoder['<|endoftext|>']]
                message_str += '<eos>'
                message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)


            # print("First encode the text to a bit sequence!")
            # print(message)  # the binary stream. text--arithmetic-->binary stream
            # print("the length is {}".format(len(message)))

            # Next encode bits into cover text, using arbitrary context
            

            # 下方完成隐写算法，使用不同隐写算法将字节流嵌入进生成文本中，得到out经过GPT2的解码器得到COVERTEXT
            Hq = 0
            if mode == 'arithmetic':
                out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
            elif mode == 'huffman':
                out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, finish_sent=finish_sent)
            elif mode == 'bins':
                out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, finish_sent=finish_sent)
            elif mode == 'saac':
                out, nll, kl, words_per_bit, Hq, topk_list, case_studies = encode_saac(model, enc, message, context_tokens, device=device, temp=temp, precision=precision, topk=topk, nucleus=nucleus)
            #     add thing contains device='cuda', temp=1.0, precision=26, topk=50, nucleus=0.95.
            covertext = enc.decode(out)
            covertext_list.append(covertext)             # 将所有COVERTEXT保存到一个结构中，可供调用



            # list_for_bpw.append(1/words_per_bit)      # 用于计算参数
            # list_for_DKL.append(kl)                   # 用于计算参数
            # list_for_seq.append(temp_count)           
            # print("="*40 + " Encoding " + "="*40)

            # 打印结果，COVERTEXT，此处可以将covertext进行提取。
            print('#{} generated covertext:\n'.format(temp_count), covertext) # covertext. generated covertext that contains secret information.
            print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
            



            # -----------------------------------------------------------------------------------
            # 以下为隐写提取过程， 选择不同的隐写算法对covertext进行提取，得到字节流 MESSAGE_REC
            # Decode binary message from bits using the same arbitrary context
        
            # 下方在编写时可能会使用到，这里先注释掉，接收人将自己的名字和covertext输入进行判定。
            # input_name = input("Please input ur name:")
            # input_covertext = input("Please input the covertext:")
            # covertext = input_covertext


            if mode == 'arithmetic':
                message_rec = decode_arithmetic(model, enc, covertext, context_tokens, temp=temp, precision=precision, topk=topk)
            elif mode == 'huffman':
                message_rec = decode_huffman(model, enc, covertext, context_tokens, block_size)
            elif mode == 'bins':
                message_rec = decode_block(model, enc, covertext, context_tokens, block_size, bin2words, words2bin)
            elif mode == 'saac':
                message_rec = decode_saac(model, enc, covertext, context_tokens, device=device, temp=temp, precision=precision, topk=topk, nucleus=nucleus)

            # print("="*40 + " Recovered Message " + "="*40)
            # print(message_rec)  # binary stream extracted from stego_text.
            # print("=" * 80)
            # Finally map message bits back to original text
            
            # 对字节流进行解码操作，最终得到的reconst变量即为最终隐写提取所得，正常使用应为人名。
            if unicode_enc:
                message_rec = [bool(item) for item in message_rec]
                ba = bitarray.bitarray(message_rec)
                reconst = ba.tobytes().decode('utf-8', 'ignore')
            else:
                reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)
                # reconst = encode_arithmetic(model, enc, message_rec, message_ctx, temp=temp, precision=precision, topk=topk)
                # print("reconst[0] is", format(reconst[0]))
                reconst = enc.decode(reconst[0])
            print("The decode text is ")
            print(reconst[0:-5])                       # Decoded text. message_rec --arithmetic decode--> reconst
            
            # 这里完成基本的判断，判断此时的covertext是否指向此人名，这里对应输入设置。
            # extracted_name = reconst.upper()[0:-5]
            # if extracted_name is input_name.upper():
            #     print("YOU ARE THE ONE! (^..^)")
            # else:
            #     print("PITY. ('..') ")





        # dataframe = pd.DataFrame({'Times':list_for_seq, 'Dkl':list_for_DKL, 'Bits/Word':list_for_bpw})
        # dataframe.to_csv("test_{}_temp_{}_topk_{}_prec_{}_nucleus_{:.3}.csv".format(mode, temp, topk, precision, nucleus), index=False, sep=',')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-unicode_enc", type=bool, default=False, help="Whether open unicode encoding method.")
    parser.add_argument("-mode", type=str, default="saac", help="Steganography Method.")
    parser.add_argument("-block_size", type=int, default=3, help="Block_size is for Huffman and Bins.")
    parser.add_argument("-temp", type=float, default=0.9, help="Temperature, for arithmetic and saac.")
    parser.add_argument("-precision", type=int, default=26, help="Precision is for arithmetic and saac.")
    parser.add_argument("-topk", type=int, default=300, help="top K Token, for arithmetic and saac.")
    parser.add_argument("-nucleus", type=float, default=0.95, help="Nucleus is for saac.")
    parser.add_argument("-device", type=str, default="cuda", help="The basic calculator when applying model.")
    parser.add_argument("-finish_sent", type=bool, default=False, help="")
    parser.add_argument("-delta", type=float, default=0.01, help="delta for adaptive arithemtic encoding method.")
    parser.add_argument("-language_model", type=str, default="gpt2", help="Basic Languages to generate text.")
    parser.add_argument("-context_file", type=str, default="./context.txt", help="the basic context file")
    parser.add_argument("-name", type=str, default="Gogo", help="Name, plz.")
    args = parser.parse_args()
    # main()
    main(args)














#     parser = argparse.ArgumentParser()
#     parser.add_argument("-plaintext", type=str, default="", help="your secret plaintext, use a double-quotes if necessary")
#     parser.add_argument("-context", type=str, default="", help="context used for steganography, use a double-quotes if necessary")
#     parser.add_argument("-encrypt", type=str, default="arithmetic", choices=["arithmetic", "utf8"])
#     parser.add_argument("-encode", type=str, default="bins", choices=["bins", "huffman", "arithmetic", "saac"])
#     parser.add_argument("-lm", type=str, default="gpt2")
#     parser.add_argument("-device", type=str, default="0", help="your gpu device id")
#     parser.add_argument("-block_size", type=int, default=4, help="block_size for bin/huffman encoding method")
#     parser.add_argument("-precision", type=int, default=26, help="precision for arithmetic encoding method")
#     parser.add_argument("-temp", type=float, default=1.0, help="temperature for arithemtic/huffman encoding method")
#     parser.add_argument("-topK", type=int, default=50, help="topK for arithemtic encoding method")
#     parser.add_argument("-nucleus", type=float, default=0.95, help="neclues for adaptive arithemtic encoding method")
#     parser.add_argument("-delta", type=float, default=0.01, help="delta for adaptive arithemtic encoding method")
#     args = parser.parse_args()
#     main(args)

# basic parameters include unicode_enc, mode, block_size, temp, precision, sample_tokens, topk, device, finish_sent, nucleus


# 12.30, fulfil the basic function api for further implementation.









# result:
# bins:
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# arithmetic:
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# huffman:
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 第一处：message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)
# 第二处：out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
# 前一个：message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
# 后一个：reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)


# [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
import random




# -----------------------------------------------------
# |       Harvard NLP project edited by Kieran        |
# | Feature SAAC, Arithmetic, Bins, Huffman implemen- |
# | tations on linguistic Steganography based on Text |
# | Generation Language Model. The Basic openAI GPT-2 |
# | language model have been included in the directo- |
# | ry pretrained_model. Usage mentioned below.       |
# -----------------------------------------------------

# Usage :
# python run_single.py [-mode] [-unicode_enc] [-block_size] [-temp] [-precision] [-topk] [-nucleus] [-device] [-finish_sent] [-delta] [-language_model]

# Simply Usage:
# python run_single.py
# python run_single.py -mode "huffman"
# python run_single.py -mode "saac" -nucleus 0.98

# API likely:
# message_str: string to be hidden.
# context: the context related to the text generation procedure.
# message: Binary stream Based on message_str. text --arithmetic encode--> binary stream
# text: covertext. generated text that contains secret information.
# message_rec: binary stream extracted from stego_text.
# reconst: Decoded text. message_rec --arithmetic decode--> reconst

# env: Windows 10, python 3.6.12, torch 1.0.1, pytorch_transformers 1.1.0,
#      bitarray 1.0.1, CUDA 10, GTX1050.

# Encrypt function is to generate 10 random encrypted word.

def encrypt(unicode_enc, mode, block_size, temp, precision, topk, device, finish_sent, model_name, delta, context, message_str):
    print("loading GPT-2 LM to GPU")
    enc, model = get_model(model_name=model_name)
    print("finish loading !")

    print("implication of {}".format(mode))
    if mode == 'bins':
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)

    if delta and mode == "saac":
        nucleus = 2 ** (-1.0 * delta)
    
    first_flag = 1
    context_tokens = encode_context(context, enc)
    while(1):
        sentence_assmble = []
        if first_flag == 0:
            message_str = input("Please reenter a new plaintext:")
            # output_amount = len(message_str)
        message_str = message_str.upper()
        arr=list(message_str)
        generated_array = dfs(arr,0,[])
        first_flag = 0
        for temp_count in range(0, len(generated_array)):
            # First encode message to uniform bits, without any context
            # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
            
            # if temp_count > 10: # protect from running too much times.
            #     break
            
            print("="*80)
            print("Altering the #{} msg_str:".format(temp_count), message_str)
            message_str = generated_array[temp_count]

            if unicode_enc:
                ba = bitarray.bitarray()
                ba.frombytes(message_str.encode('utf-8'))
                message = ba.tolist()
            else:
                message_ctx = [enc.encoder['<|endoftext|>']]
                message_str += '<eos>'
                message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)
                # message = decode_arithmetic(model, enc, message_str, message_ctx, precision=precision, topk=topk, temp=temp)

            Hq = 0
            if mode == 'arithmetic':
                out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
            elif mode == 'huffman':
                out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, finish_sent=finish_sent)
            elif mode == 'bins':
                out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, finish_sent=finish_sent)
                words_per_bit = 1
            elif mode == 'saac':
                out, nll, kl, words_per_bit, Hq, topk_list, case_studies = encode_saac(model, enc, message, context_tokens, device=device, temp=temp, precision=precision, topk=topk, nucleus=nucleus)
            #     add thing contains device='cuda', temp=1.0, precision=26, topk=50, nucleus=0.95.
            text = enc.decode(out)
            # print("="*40 + " Encoding " + "="*40)
            print('#{} generated covertext:\n'.format(temp_count), text) # covertext. generated text that contains secret information.
            # print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
            sentence_assmble.append(text)
        dataframe = pd.DataFrame({'Sentences':sentence_assmble})
        dataframe.to_csv("User_{}_Name_{}_Amount_{}.csv".format(random.randint(1,10000), message_str.upper()[0:-5], len(generated_array)), index=False, sep=',')








def main(args):
    # Initial process
    args = vars(args)
    unicode_enc = args['unicode_enc']
    mode = args['mode']
    block_size = args['block_size']
    temp = args['temp']
    precision = args['precision']
    topk = args['topk']
    device = args['device']
    finish_sent = args['finish_sent']
    nucleus = args['nucleus']
    delta = args['delta']
    model_name = args['language_model']
    context_file = args['context_file']
    sample_tokens = 100

    # PARAMETERS
    message_str = "Joe"  # string to be hidden.

    # VALIDATE PARAMETERS
    if mode not in ['arithmetic', 'huffman', 'bins', 'sample', 'saac']:
        raise NotImplementedError
    
    print("Default plain_text is ", message_str)
    # context = """Whereas traditional cryptography encrypts a secret message into an unintelligible form, steganography conceals that communication is taking place by encoding a secret message into a cover signal. Language is a particularly pragmatic cover signal due to its benign occurrence and independence from any one medium. Traditionally, linguistic steganography systems encode secret messages in existing text via synonym substitution or word order rearrangements. Advances in neural language models enable previously impractical generation-based techniques. We propose a steganography technique based on arithmetic coding with large-scale neural language models. We find that our approach can generate realistic looking cover sentences as evaluated by humans, while at the same time preserving security by matching the cover message distribution with the language model distribution.""" 
    f = open(context_file, 'r', encoding='utf-8')
    context = f.read()
    f.close()
    print("sample context is ", context)   # related to the text generation procedure.
    encrypt(unicode_enc, mode, block_size, temp, precision, topk, device, finish_sent, model_name, delta, context, message_str)




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
    args = parser.parse_args()
    # main()
    main(args)

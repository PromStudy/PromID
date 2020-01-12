#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import sys
import re
import math
import time
import argparse
import logging
import bisect
from pkg_resources import resource_filename
 

#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.0')
    parser.add_argument('-I', metavar='input', required=True,
                        help='path to the input genome')
    parser.add_argument('-O', metavar='output', required=True,
                        help='path to the output file')    
    parser.add_argument('-D', metavar='distance', default=500,
                        type=int, choices=range(50, 10000),
                        help='minimum soft distance between the predicted TSS '
                             ', defaults to 500')
    parser.add_argument('-T', metavar='threshold', default=0.5,
                        type=float, 
                        help='decision threshold for the prediction model'
                             ', defaults to 0.5')
    parser.add_argument('-C', metavar='chromosomes', default="",
                        type=str, help='comma separated list of chromosomes to use for promoter prediction '
                             ', defaults to all chromosomes')

    args = parser.parse_args()

    return args

def encode(ns, strand):
    if(strand == "+"):
        rep = {"A": "1,0,0,0,", "T": "0,1,0,0,", "G": "0,0,1,0,", "C": "0,0,0,1,", "N": "0,0,0,0,"} 
    else:
        ns = ns[::-1]
        rep = {"A": "0,1,0,0,", "T": "1,0,0,0,", "G":"0,0,0,1," , "C": "0,0,1,0,", "N": "0,0,0,0,"} 
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    ns = pattern.sub(lambda m: rep[re.escape(m.group(0))], ns)
    return np.fromstring(ns[:-1], dtype=int, sep=",").reshape(-1, 4) 

def close(s, a):
    fmd = float('inf')
    for v in a:
        if(abs(s - v) < fmd):
            fmd = abs(s - v)
    return fmd

def clean_seq(s):
    ns = s.upper()    
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def pick(scores, dt, minDist, strand):
    all_chosen = []
    rows = []
    for s in range(len(scores)):
        tss = scores[s][0]
        score = scores[s][1]
        scaling = 1.0
        if(len(all_chosen) > 0):
            fmd = abs(tss - find_nearest(all_chosen, tss))
            if(fmd < minDist):
                scaling = fmd / minDist
        if(score*scaling >= dt):
            bisect.insort(all_chosen, tss)
            rows.append([tss, score, strand])
    return rows

def pick_scan(scores, dt, minDist):
    all_chosen = []
    for s in range(len(scores)):
        tss = scores[s][0]
        score = scores[s][1]
        scaling = 1.0
        if(len(all_chosen) > 0):
            fmd = abs(tss - find_nearest(all_chosen, tss))
            if(fmd < minDist):
                scaling = fmd / minDist
        if(score*scaling >= dt):
            bisect.insort(all_chosen, tss)
    return all_chosen


def main():    
    args = get_options()

    if None in [args.I, args.O]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -D distance -T1 threshold1 '
                      '-T2 threshold2 -C chromosomes')
        exit()

    print("PromID 1.01")
    sLen = 1001
    half_size = 500
    batch_size = 128
    dt1 = 0.1
    dt2 = args.T
    minDist = args.D 
    #print("Scan threshold: " + str(dt1))
    print("Prediction threshold: " + str(dt2))
    print("Parsing fasta at " + str(args.I))
    fasta = {}
    seq = ""
    good_chr = []
    if(args.C!=""):
        good_chr = args.C.split(",")
    with open(args.I) as f:
        for line in f:
            if(line.startswith(">")):
                if(len(seq)!=0):
                    if(chrn in good_chr or args.C==""):
                        seq = clean_seq(seq)
                        fasta[chrn] = seq
                        print(chrn + " - " + str(len(seq)))   
                chrn = line.strip()[1:]             
                seq = ""
                continue                
            else:
                seq+=line
        if(len(seq)!=0):
            if(chrn in good_chr or args.C==""):
                seq = clean_seq(seq)
                fasta[chrn] = seq
                print(chrn + " - " + str(len(seq))) 

    putative = {}
    scan_step = 100
    print("")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("")  
    new_graph = tf.Graph()
    with tf.compat.v1.Session(graph=new_graph) as sess:
        tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], resource_filename(__name__, "models/model_scan"))
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, resource_filename(__name__, "models/model_scan/variables/variables") )
        input_x = tf.compat.v1.get_default_graph().get_tensor_by_name("input_prom:0")
        y = tf.compat.v1.get_default_graph().get_tensor_by_name("output_prom:0")
        kr = tf.compat.v1.get_default_graph().get_tensor_by_name("kr:0")
        in_training_mode = tf.compat.v1.get_default_graph().get_tensor_by_name("in_training_mode:0")    
        for key in fasta.keys(): 
            print("Scanning " + key)
            for strand in ["+", "-"]:
                ck = key + strand
                putative[ck] = []
                j = half_size
                m = 1
                batch = []
                inds = []
                while(j < len(fasta[key]) - half_size - 1):
                    fa = fasta[key][j - half_size: j + half_size + 1]
                    if(len(fa) == sLen):   
                        batch.append(encode(fa, strand))
                        inds.append(j)
                    if(len(batch) >= batch_size or j + scan_step >= len(fasta[key]) - half_size - 1):
                        predict = sess.run(y, feed_dict={input_x: batch, kr: 1.0, in_training_mode: False})
                        chosen = [[inds[i], predict[i][0]] for i in range(len(batch)) if  predict[i][0] > dt1]
                        putative[ck].extend(chosen)  
                        batch = []
                        inds = []         
                    j = j + scan_step
                    if(j > m * 10000000):
                        print(str(j))
                        m = m + 1

                putative[ck].sort(key=lambda x: x[1], reverse=True)
                putative[ck] = pick_scan(putative[ck], dt1, minDist)
                putative[ck].sort()
                print("Scanned " + strand + " strand. Found " + str(len(putative[ck])) + " promoter regions.")

    out = [] 
    new_graph = tf.Graph()
    with tf.compat.v1.Session(graph=new_graph) as sess:
        tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], resource_filename(__name__, "models/model_pos") )
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, resource_filename(__name__, "models/model_pos/variables/variables") )
        input_x = tf.compat.v1.get_default_graph().get_tensor_by_name("input_prom:0")
        y = tf.compat.v1.get_default_graph().get_tensor_by_name("output_prom:0")
        kr = tf.compat.v1.get_default_graph().get_tensor_by_name("kr:0")
        in_training_mode = tf.compat.v1.get_default_graph().get_tensor_by_name("in_training_mode:0")    
        for key in fasta.keys(): 
            print("Predicting " + key)
            rows = []
            for strand in ["+", "-"]:    
                scores = []
                ck = key + strand
                m = 1
                for p in putative[ck]:
                    batch = []
                    inds = []
                    for j in range(p - int(scan_step/2), p + int(scan_step/2) + 1):
                        fa = fasta[key][j - half_size: j + half_size + 1]
                        if(len(fa) == sLen):   
                            batch.append(encode(fa, strand))
                            inds.append(j)
                    predict = sess.run(y, feed_dict={input_x: batch, kr: 1.0, in_training_mode: False})
                    mi = predict.argmax(axis=0)[0]
                    scores.append([inds[mi], predict[mi][0]])                        
                    if(p > m * 10000000):
                        print(str(p))
                        m = m + 1

                scores.sort(key=lambda x: x[1], reverse=True)
                new_scores = pick(scores, dt2, minDist, strand)
                rows.extend(new_scores)
                print("Predicted " + strand + " strand. Found " + str(len(new_scores)) + " promoters.")

            rows.sort(key=lambda x: x[0])
            for row in rows:
                col = "255,0,0"
                if(row[2]=="-"):
                    col = "0,0,255"
                out.append(key + "\t" + str(row[0]-half_size + 1) + "\t" + str(row[0] + half_size + 2) + "\t" +
                    key+":"+str(row[0] - half_size + 1)+":" + str(row[0] + half_size + 2)+":"+row[2]+"\t" +
                str(row[1]) +"\t" + row[2] +"\t" + str(row[0])+"\t" +str(row[0] + 1)+"\t" +col)

    with open(args.O, 'w+') as f:
        f.write('\n'.join(out))


if __name__ == '__main__':
    main()
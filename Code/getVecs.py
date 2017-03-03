#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess sentences for Word2Vec and Prouctivity Analysis.
"""
__author__ = "Nick Kloehn"
__copyright__ = "Copyright 2017, Nick Kloehn"
__credits__ = []
__version__ = "1.0"
__maintainer__ = "Nick Kloehn"
__email__ = "See the author's website"

################################################################################
# Imports and Globals
from affixes import *
import numpy as np
from collections import Counter
from nltk.stem import *

f = ('/Users/pokea/Documents/Work/UofA/'
     'Current/Morph2Vec/Glove/glove.6B/glove.6B.200d.txt')

foutp = ('/Users/pokea/Documents/Work/UofA/'
        'Current/Morph2Vec/Shared/morphvecs_prefixes.txt')

fouts = ('/Users/pokea/Documents/Work/UofA/'
        'Current/Morph2Vec/Shared/morphvecs_suffixes.txt')

foutp2 = ('/Users/pokea/Documents/Work/UofA/'
        'Current/Morph2Vec/Shared/prefixes.txt')

fouts2 = ('/Users/pokea/Documents/Work/UofA/'
        'Current/Morph2Vec/Shared/suffixes.txt')

stemmer = PorterStemmer()
Lex = Counter()
pTypes = dict()
sTypes = dict()

################################################################################

def getUnderived(string,affix):
        if affix in string:
            return string.replace(affix,"")
        else:
            return None

def writeout(myDict,f1,f2):
    with open(f1, 'w') as out1:
        with open(f2,'w') as out2:
            for k,v in pTypes.items():
                strlist = str()
                for vec in v[1]:
                    strlist += str(vec) + ' '
                    string = str(v[0]) + "\t" + k + "\t" + strlist + "\n"
                    out1.write(string)
                out2.write(k + "\n")


with open(f, encoding="ISO-8859-1") as infile:
    #go through files and add all words to FreqDist
    for line in infile:
        values = line.split()
        word = values[0]
        Lex[word] = 1

with open(f, encoding="ISO-8859-1") as infile:    
    for line in infile:
        values = line.split()
        derived = values[0]
        stem = stemmer.stem(derived)
        for prefix in prefixes:
            if derived.startswith(prefix):
                try:
                    underived = getUnderived(derived,prefix)
                    if underived in stem:
                        if underived in Lex:
                            coefs = np.asarray(values[1:], dtype='float32')
                            pTypes[derived] = (prefix,coefs)
                except:
                    continue
                
        for suffix in suffixes:
            if derived.endswith(suffix):
                try:
                    underived = getUnderived(derived,suffix)
                    if underived in  stem:
                        if underived in Lex:
                            coefs = np.asarray(values[1:], dtype='float32')
                            sTypes[derived] = (suffix,coefs)
                except:
                    continue

writeout(pTypes,foutp,foutp2)
writeout(sTypes,fouts,fouts2)


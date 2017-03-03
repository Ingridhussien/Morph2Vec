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
# Imports
from affixes import *
import re
import numpy as np
from collections import Counter
from nltk.stem import *
from nltk.corpus import wordnet as wn
################################################################################

def lookup(word):
    """Returns the Number of Definitions of a word from Wordnet."""
    return len(wn.synsets(word))

def cleanString(string):
    return re.compile('[\W_]+').sub('', string)

def getUnderived(string,affix,front):
    for m in re.compile(affix).finditer(string):
        if front:
            return string[m.end():]
        else:
            return string[:m.start()]

class Morphs:

    def __init__(self):
        self.outdicts = [dict() for n in range(len(affixes))]
        self.stemmer = PorterStemmer()
        pass

    def getLex(self,f):
        """Generate a lexicon of all words, with the vectors as values"""
        self.Lex = dict()
        with open(f, encoding="ISO-8859-1") as file:
            #go through files and add all words to FreqDist
            for line in file:
                vecData = line.split()
                word = vecData[0]
                self.Lex[word] = self.stemmer.stem(word)
        print("There are " + str(len(self.Lex.items())) + " items in the Lexicon!") 

    def getMorphPairs(self,f):
        """Get the underived,derived paris for each affix"""
        with open(f, encoding="ISO-8859-1") as file:
            for line in file:
                values = line.split()
                word = values[0]
                for idx in range(len(affixes)):
                    affix = affixes[idx]
                    if affix in prefixes:
                        if word.startswith(affix):
                            underived = getUnderived(word,affix,True)
                            if underived:
                                # strip non word chars
                                underived = cleanString(underived)
                                # if underived in our lexicon
                                if underived in self.Lex:
                                    self.outdicts[idx][word] = underived

                    elif affix in suffixes:
                        if word.endswith(affix):
                            underived = getUnderived(word,affix,False)
                            if underived:
                                # strip non word chars
                                underived = cleanString(underived)
                                # if underived in our lexicon
                                if underived in self.Lex:
                                    self.outdicts[idx][word] = underived


    def writePairs(self,path):
        """Write out the dictionaries to files!"""
        self.outfs = [path + affix for affix in affixes]
        for idx in range(len(self.outdicts)):
            myDict = self.outdicts[idx]
            with open(self.outfs[idx], 'w') as f:
                for k,v in myDict.items():
                    string = k + "\t" + v + "\n"
                    f.write(string)


test = Morphs()
test.getLex('/Users/pokea/Documents/Work/UofA/Current/Morph2Vec/Glove/glove.6B/glove.6B.200d.txt')
test.getMorphPairs('/Users/pokea/Documents/Work/UofA/Current/Morph2Vec/Glove/glove.6B/glove.6B.200d.txt')
test.writePairs('/Users/pokea/Documents/Work/UofA/Current/Morph2Vec/Shared/Pairs/')

"""
Get frequency counts for various morphological types, and compute
measures correlated with class productivity.
"""
__author__ = "Nick Kloehn"
__copyright__ = "Copyright 2016, Nick Kloehn"
__credits__ = []
__version__ = "1.0"
__maintainer__ = "Nick Kloehn"
__email__ = "See the author's website"

################################################################################
# Imports and Globals
################################################################################
from affixes import *
from collections import Counter
from os import walk
import math, numpy as np, os, re, rpy2.robjects as robjects
from nltk.corpus import wordnet as wn
from nltk.stem import *
from collections import Counter
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from google_ngram_downloader import readline_google_store
r = robjects.r
rbase = importr("robustbase")
stemmer = PorterStemmer()
# location of corpus
path = ('/Users/pokea/Documents/Work/UofA/Current'
        '/Gentropy/WaCKyCorpus')
################################################################################
# Helper Function
################################################################################

def lookupcounts(word):
    """Go through all words in the google corpus and lookup frequency counts."""
    count = 0
    fname, url, records = next(readline_google_store(ngram_len=1, indices=word[0]))
 
    try:
        record = next(records)
 
        while record.ngram != word:
            record = next(records)
 
        while record.ngram == word:
            count = count + record.match_count
            record = next(records)
        
    except StopIteration:
        pass


def lookup(word):
    """Returns the Number of Definitions of a word from Wordnet."""
    return len(wn.synsets(word))
    
def get_ratio(x,y):
    """Calculate the number of types above line / all types. Also,
    calculate the derived frequency of all points above line over all
    derived frequencies."""
    aboveType, allType, aboveTok, allTok = 0,0,0,0
    for i in range(len(x)):
        allType += 1
        allTok += x[i]
        if y[i] >= x[i]:
            aboveTok += x[i]
            aboveType += 1
    return float(aboveType)/allType, float(aboveTok)/allTok
    
def LTS(x,y):
    """Do a Lest Trimmed Squares Analysis of the surface and base
    frequencies. Calls R, and calls the robustbase package. Returns
    all these different variables associated with productivity."""
    # Get token and type PR
    typePR, tokenPR = get_ratio(x,y)
    #
    # R Coding!
    #
    x = robjects.FloatVector(x)
    y = robjects.FloatVector(y)
    # ... and put them in the global environment
    robjects.globalenv["x"] = x
    robjects.globalenv["y"] = y
    # Create model
    robjects.globalenv["model"] = r('ltsReg(x,y)')
    intercept = r('model$intercept')[0]
    # Spearmans Rho and significance
    rho = r('cor.test(x, y, method="spearman")')
    robjects.globalenv["rho"] = rho
    prob = float(r('rho$p.value')[0])
    corr = float(r('rho$estimate')[0])
    #
    # R Coding complete!
    #
    return [corr, prob, intercept, typePR, tokenPR]

def extractVars(typeSet, vocab):
    """"Take all the matches of a morphological type and get the
    frequencies of the word/base pairs. Compare them to figure out the
    different vars. Also get # of dictionary entries from wordnet."""
    # Init token count, Hapax Count, and calculate Type Count
    N, V1, typeDefs, V  = 0, 0, 0, len(typeSet)
    # Init frequency structures for r-squared
    x, y = [], []
    # Go through types
    for word,base in typeSet:
        typeDefs += lookup(word)
        # get word token frequency
        freqWord = vocab[word]
        # ... and add it to total token frequence (n)
        N += freqWord
        # If it's a hapax,
        if freqWord == 1:
            # add it to V1
            V1 += 1
        # Get log frequency for r-squared
        logfreqWord = math.log(freqWord)
        # Check whether base exists
        if base in vocab.keys():
            logfreqBase = math.log(vocab[base])
        else:
            logfreqBase = float(0)
        #Add frequencies to x and y
        x.append(logfreqWord)
        y.append(logfreqBase)
    # Calculate P by dividing P1 by N
    try: P = float(V1)/n
    except: P = 0
    # Get Type definition average
    try: avgDefs = float(typeDefs)/ len(typeSet)
    except: avgDefs = 0
    try:
        # Do r-squared and extract results
        return [V1, P, V] + LTS(x,y) + [avgDefs]
    except:
        return [V1, P, V] + [None]*5 + [avgDefs]
    
def stripTypes(word, affix, front):
    """Strip off the morpheme from a complex form, and return both the
    complex form and the base, if both forms share the same lemma."""

    lemma = stemmer.stem(word)
    for m in re.compile(affix).finditer(word):
        if front:
            base = word[m.end():]
        else:
            base = word[:m.start()]
    if base in lemma:
        print("Affix: " + affix)
        print("Derived: " + word)
        print("Underived: " + base)
        print("\n")
        return base
    else:
        return False
    

def addTypes(vocabulary, affixes, mTypes, front):
    """For the affix keys, add all derived types, and get their
    underived forms as values."""
    for affix in affixes:
        # Value for each affix will be a set containing tuples of der/under forms
        mTypes[affix] = set()
        # cylce through the Corpus...   
        for word in vocabulary.keys():
            # if prefixes,
            if front:
                if word.startswith(affix):
                    # match and then get the value by stripping from front,
                    base = stripTypes(word, affix, True)
                    if base in vocabulary.keys():
                        print(word)
                        print(base)
                        print("\n")
                        mTypes[affix].add((word,base))
            else:
                # and if suffix,
                if word.endswith(affix):
                    # match and get the value by stripping from the back
                    base = stripTypes(word, affix, False)
                    if base in vocabulary.keys():
                        print(word)
                        print(base)
                        print("\n")
                        mTypes[affix].add((word,base))
    return mTypes


class Count:
    
    def corpus(self,path):

        self.path = path
        self.corpus = list()
        self.outf = path + '/AffixPairs'
        self.outf2 = path + '/Predictors'
        self.results = {}
        self.vocab = Counter()
        self.affixes = (prefixes,suffixes)
            
        for (dirpath, dirnames, fs) in walk(self.path):
                
            self.corpus.extend([dirpath + '/' + f for f in fs if 'UK' in f])
                
        # parse corpus files and record counts of tokens
        for f in self.corpus:
            if f.endswith('.xml'):
                with open(f, encoding="ISO-8859-1") as infile:
                    for line in infile:
                        try:
                            tok, tag, lemma = re.split('\t+', line.strip())
                            token = tok.lower()
                            self.vocab[token] += 1
                        except:
                            continue
  
        # Go through the dictionary we just created,  and match words to the classes and
        # add those words to a set that is the value of the
        # morpheme (key) in a dict
        pfixes, sfixes = self.affixes
        self.mTypes = addTypes(self.vocab,pfixes,mTypes={},front=True)
        self.mTypes = addTypes(self.vocab,sfixes,mTypes=self.mTypes,front=False)
        # Go through all affixes, and get variables (found in Readme)
        for morph in self.mTypes.keys():
            self.results[morph] = extractVars( self.mTypes[morph], self.vocab)

        with open(self.outf, 'w') as fout:
            for affix,wordset in self.mTypes.items():
                fout.write(affix)
                fout.write("\t")
                fout.write(str(wordset))
                fout.write("\n")
                
        with open(self.outf2, 'w') as fout:
            for affix,resulist in self.results.items():
                fout.write(affix)
                fout.write("\t")
                fout.write(str(resulist))
                fout.write("\n")                
            
                        
Count().corpus(path)

"""
Get frequency counts for various morphological types from Google N-gram Corpus, and compute
measures correlated with class productivity.
"""
__author__ = "Nick Kloehn"
__copyright__ = "Copyright 2017, Nick Kloehn"
__credits__ = []
__version__ = "2.0"
__maintainer__ = "Nick Kloehn"
__email__ = "See the author's website"

################################################################################
# Imports and Globals
################################################################################
from collections import Counter
from os import walk
import math, numpy as np, os, re, rpy2.robjects as robjects
from nltk.corpus import wordnet as wn
from collections import Counter
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from google_ngram_downloader import readline_google_store
import pdb
r = robjects.r
rbase = importr("robustbase")
################################################################################
# Helper Function
################################################################################

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

def extractVars(pairDict,vocab):
    """"Take all the matches of a morphological type and get the
    frequencies of the word/base pairs. Compare them to figure out the
    different vars. Also get # of dictionary entries from wordnet."""
    # Init token count, Hapax Count, and calculate Type Count
    N, V1, typeDefs, V  = 0, 0, 0, len(pairDict.items())
    # Init frequency structures for r-squared
    x, y = [], []
    # Go through types
    for derived,underived in pairDict.items():
        typeDefs += lookup(derived)
        # get derived token frequency
        freqDer = vocab[derived]
        # ... and add it to total token frequence (n)
        N += freqDer
        # If it's a hapax,
        if freqDer == 1:
            # add it to V1
            V1 += 1
        # Get log frequency for r-squared
        logfreqDer = math.log(freqDer)
        # Check whether base exists
        if underived in vocab.keys():
            logfreqUnd = math.log(vocab[underived])
        else:
            logfreqBase = float(0)
        #Add frequencies to x and y
        x.append(logfreqDer)
        y.append(logfreqUnd)
    # Calculate P by dividing P1 by N
    try: P = float(V1)/n
    except: P = 0
    # Get Type definition average
    try: avgDefs = float(typeDefs)/ len(pairDict.items())
    except: avgDefs = 0
    try:
        # Do r-squared and extract results
        return [V1, P, V] + LTS(x,y) + [avgDefs]
    except:
        return [V1, P, V] + [None]*5 + [avgDefs]

class Corpora:
    """Obeject to contain the corpus files."""

    def __init__(self,cDir):
        """Get the names of the files."""
        self.theFiles = list()
        self.Ngrams = Counter()

        # get all file names/paths
        for (dirpath, dirnames, fs) in walk(cDir):
            # Got through each edited pair file
            for f in fs:
                if f != '.DS_Store':
                    info = (dirpath + '/' + f,f)
                    self.theFiles.append(info)
            self.dir = dirpath

        # Go through the files and create a dict for each file. Add the word pairs to the dict.
        self.affixDicts = [Counter()for n in range(len(self.theFiles))]
        self.affixNames = [f[1] for f in self.theFiles]
        for idx in range(len(self.theFiles)):
            with open(self.theFiles[idx][0], encoding="ISO-8859-1") as infile:
                for line in infile:
                    d,u = re.split('\t',line.strip())
                    self.affixDicts[idx][d] = u
                    self.Ngrams[d],self.Ngrams[u] = 0,0

    def getAffixDicts(self):
        return self.affixDicts

    def getAffixNames(self):
        return self.affixNames

    def getPath(self):
        return self.dir

    def getNgrams(self):
        """Get Frequency of Words in Google Ngram Corpus."""
        keys = self.Ngrams.keys()
        for char in list(map(chr, range(97, 123))):
                try:
                    fname, url, records = next(readline_google_store(ngram_len=1,indices=char))
                    count = 0
                    current = ''
                    while records:
                        try:
                            token,year,match,volume = next(records)
                            #print(token)
                            if token in keys:
                                if token == current:
                                    count += match
                                else:
                                    self.Ngrams[current] = count
                                    current = token
                                    count = 0
                            else:
                                continue
                        except StopIteration:
                            break
                except:
                    continue
        return self.Ngrams


class Count:

    def __init__(self,path):
        """Initialize object by getting the organized corpus files from Corpora object"""
        c = Corpora(path)
        self.affixDicts = c.getAffixDicts()
        self.affixNames = c.getAffixNames()
        self.outf = c.getPath() + '/Predictors'
        self.Ngrams = c.getNgrams()
        self.results = dict()

    def getRatios(self):
        """Got through affix pairs and calculate the productivity ratios."""
        for idx in range(len(self.affixNames)):
            affixName = self.affixNames[idx]
            affixDict = self.affixDicts[idx]
            self.results[affixName] = extractVars(affixDict,self.Ngrams)

    def writeVars(self):
        with open(self.outf, 'w') as fout:
            for affix,resulist in self.results.items():
                fout.write(affix)
                fout.write("\t")
                fout.write(str(resulist))
                fout.write("\n")                
            
def main():              
    test = Count('/Users/pokea/Documents/Work/UofA/Current/Dissertation/Morph2Vec/Morph2Vec/EditedPairs')
    test.getRatios()
    test.writeVars()

if __name__ == '__main__':
    main()

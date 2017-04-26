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
import re
from os import walk
import math
from google_ngram_downloader import readline_google_store
################################################################################
# Helper Function
################################################################################

class Corpora:
    """Obeject to contain the corpus files."""

    def __init__(self,cDir):
        """Get the names of the files."""
        self.path = cDir
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

    def getNgrams(self):
        """Get Frequency of Words in Google Ngram Corpus."""
        keys = self.Ngrams.keys()
        alphabet = list(map(chr, range(97, 123)))
        count = 0
        current = ''
        for char in alphabet:
            googleGen = readline_google_store(ngram_len=1,indices=char)
            while googleGen:
                try:
                    name, url, wordGen = next(googleGen)
                    while wordGen:
                        try:
                            token,year,match,volume = next(wordGen)
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
                except StopIteration:
                    break
            print("Finished with"+"\t"+char+"\n")
        print("Ngram Counts Completed!")

    def writeOut(self):
        outf = self.path + 'Dictionary'
        with open(outf, 'w') as fout:
            for token,count in self.Ngrams.items():
                fout.write(token)
                fout.write("\t")
                fout.write(str(count))
                fout.write("\n")          
            
def main():              
    test = Corpora('/Users/pokea/Documents/Work/UofA/Current/Dissertation/Morph2Vec/Morph2Vec/EditedPairs')
    test.getNgrams()
    test.writeOut()

if __name__ == '__main__':
    main()

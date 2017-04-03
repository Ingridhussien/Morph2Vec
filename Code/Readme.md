affixes.py contains a python list of all the affixes tested
getMorphs.py goes through the glove vectors and matches all underived/derived pairs of each morpheme and writes them to file
vecAnalysis.py takes all the edited pairs and calculates the average cosine similarity to the average vector of each class, and their standard deviations
getRatios.py calculates the frequency ratios of the affixes
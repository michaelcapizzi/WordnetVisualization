__author__ = 'mcapizzi'

from nltk import *
import numpy as np
import gensim
from nltk.corpus import wordnet
import accessWordnet as nl
from sklearn import *
from sklearn.cluster import KMeans

from sklearn.manifold import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import sys

#   Takes two arguments:
#       word to analyze
#       type of dimension reduction ("p" = PCA, "t" = TSNE)

# TODO - handle if there is not data in the relation type
# TODO - add parameter for types of wordnet relationships to include

# access word2Vec model
model = gensim.models.word2vec.Word2Vec.load_word2vec_format("/home/mcapizzi/Google_Drive_Arizona/Programming/word2Vec/GoogleNews-vectors-negative300.bin.gz", binary=True)

###################
# sim = model.most_similar(positive=sys.argv[1], topn=10)
###################

# get word relations for word (imported from nltk_test)
synonyms = nl.get_wn_relation(sys.argv[1], "s")
antonyms = nl.get_wn_relation(sys.argv[1], "a")
hypernyms = nl.get_wn_relation(sys.argv[1], "hr")
hyponyms = nl.get_wn_relation(sys.argv[1], "ho")


# builds dict of word with [0] its word2vec vector, [1] cosine similarity to word, and [2] euclidean distance from word
wordDict = {}
synWorddict = {}
antWorddict = {}
hyperWorddict = {}
hypoWorddict = {}

#TODO - make separate dict for word-to-analyze so that it can be different shape.color
#can't be done as separate dataset - because PCA depends upon other values to compare to?
# add word to worddict (cosine similarity and euclidean distance will be preset)
synWorddict[sys.argv[1]] = (model[sys.argv[1]], 1.0, 0.0)
#wordDict[sys.argv[1]] = (model[sys.argv[1]], 1.0, 0.0)

#build dictionaries
for item in synonyms:
    if item in model:
        synWorddict[item] = (model[item], model.similarity(sys.argv[1], item), euclidean_distance(model[item], model[sys.argv[1]]))
    else:
        continue

for item in antonyms:
    if item in model:
        antWorddict[item] = (model[item], model.similarity(sys.argv[1], item), euclidean_distance(model[item], model[sys.argv[1]]))
    else:
        continue

for item in hypernyms:
    if item in model:
        hyperWorddict[item] = (model[item], model.similarity(sys.argv[1], item), euclidean_distance(model[item], model[sys.argv[1]]))
    else:
        continue

for item in hyponyms:
    if item in model:
        hypoWorddict[item] = (model[item], model.similarity(sys.argv[1], item), euclidean_distance(model[item], model[sys.argv[1]]))
    else:
        continue

# ######word
#
# word_data = np.array(model[sys.argv[1]])
#
# if sys.argv[2] == "p":                                      # PCA - reduce to 2 dimensions
#     word_pca = decomposition.PCA(n_components=2)
#     word_output = np.array(word_pca.fit_transform(word_data))
# elif sys.argv[2] == "t":
#     word_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
#     word_output = np.array(word_tsne.fit_transform(word_data))
# else:
#     word_pca = decomposition.PCA(n_components=2)
#     word_output = np.array(word_pca.fit_transform(word_data))
#
# # get cartesian coordinates
# wordX = np.array(map(lambda z: z[0], word_output))
# wordy = np.array(map(lambda z: z[1], word_output))


######synonyms

synDictKeys = synWorddict.keys()
synDictValues = map(lambda z: synWorddict[z], synWorddict)


syn_all_data = np.array(map(lambda x: x[0], synDictValues))


if sys.argv[2] == "p":                                      # PCA - reduce to 2 dimensions
    syn_pca = decomposition.PCA(n_components=2)
    syn_output = np.array(syn_pca.fit_transform(syn_all_data))
elif sys.argv[2] == "t":
    syn_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
    syn_output = np.array(syn_tsne.fit_transform(syn_all_data))
else:
    syn_pca = decomposition.PCA(n_components=2)
    syn_output = np.array(syn_pca.fit_transform(syn_all_data))


# TODO - find way to use if/then to sort word from synonyms
# get cartesian coordinates
synX = np.array(map(lambda z: z[0], syn_output))
syny = np.array(map(lambda z: z[1], syn_output))
syn_cossim = np.array(map(lambda z: z[2], synDictValues))

######antonyms

antDictKeys = antWorddict.keys()
antDictValues = map(lambda z: antWorddict[z], antWorddict)

ant_all_data = np.array(map(lambda x: x[0], antDictValues))


if sys.argv[2] == "p":                                      # PCA - reduce to 2 dimensions
    ant_pca = decomposition.PCA(n_components=2)
    ant_output = np.array(ant_pca.fit_transform(ant_all_data))
elif sys.argv[2] == "t":
    ant_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
    ant_output = np.array(ant_tsne.fit_transform(ant_all_data))
else:
    ant_pca = decomposition.PCA(n_components=2)
    ant_output = np.array(ant_pca.fit_transform(ant_all_data))


# get cartesian coordinates
if ant_output.any():
    antX = np.array(map(lambda z: z[0], ant_output))
    anty = np.array(map(lambda z: z[1], ant_output))
    ant_cossim = np.array(map(lambda z: z[2], antDictValues))
else:
    antX = np.array([])
    anty = np.array([])
    ant_cossim = np.array([])
# cosine similarity to word


######hypernyms

hyperDictKeys = hyperWorddict.keys()
hyperDictValues = map(lambda z: hyperWorddict[z], hyperWorddict)


hyper_all_data = np.array(map(lambda x: x[0], hyperDictValues))


if sys.argv[2] == "p":                                      # PCA - reduce to 2 dimensions
    hyper_pca = decomposition.PCA(n_components=2)
    hyper_output = np.array(hyper_pca.fit_transform(hyper_all_data))
elif sys.argv[2] == "t":
    hyper_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
    hyper_output = np.array(hyper_tsne.fit_transform(hyper_all_data))
else:
    hyper_pca = decomposition.PCA(n_components=2)
    hyper_output = np.array(hyper_pca.fit_transform(hyper_all_data))


# get cartesian coordinates
if hyper_output.any():
    hyperX = np.array(map(lambda z: z[0], hyper_output))
    hypery = np.array(map(lambda z: z[1], hyper_output))
    hyper_cossim = np.array(map(lambda z: z[2], hyperDictValues))
else:
    hyperX = np.array([])
    hypery = np.array([])
    hyper_cossim = np.array([])


######hyponyms

hypoDictKeys = hypoWorddict.keys()
hypoDictValues = map(lambda z: hypoWorddict[z], hypoWorddict)


hypo_all_data = np.array(map(lambda x: x[0], hypoDictValues))


if sys.argv[2] == "p":                                      # PCA - reduce to 2 dimensions
    hypo_pca = decomposition.PCA(n_components=2)
    hypo_output = np.array(hypo_pca.fit_transform(hypo_all_data))
elif sys.argv[2] == "t":
    hypo_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
    hypo_output = np.array(hypo_tsne.fit_transform(hypo_all_data))
else:
    hypo_pca = decomposition.PCA(n_components=2)
    hypo_output = np.array(hypo_pca.fit_transform(hypo_all_data))


# get cartesian coordinates
if hypo_output.any():
    hypoX = np.array(map(lambda z: z[0], hypo_output))
    hypoy = np.array(map(lambda z: z[1], hypo_output))
    hypo_cossim = np.array(map(lambda z: z[2], hypoDictValues))
else:
    hypoX = np.array([])
    hypoy = np.array([])
    hypo_cossim = np.array([])


####### KMeans

total_data = np.concatenate((syn_output, ant_output, hyper_output, hypo_output))

#TODO - consider changing initial k parameter?
#how to set?
kmeans = KMeans(n_clusters=5)
kmeans.fit(total_data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

syn_cluster = kmeans.predict(syn_output)
print("synonyms are in the following clusters:\n")
for syn in syn_cluster:
    print(syn)

print("\n")

ant_cluster = kmeans.predict(ant_output)
print("antonyms are in the following clusters:\n")
for ant in ant_cluster:
    print(ant)

print("\n")

hyper_cluster = kmeans.predict(hyper_output)
print("hypernyms are in the following clusters:\n")
for hyp in hyper_cluster:
    print(hyp)

print("\n")

hypo_cluster = kmeans.predict(hypo_output)
print("hyponyms are in the following clusters:\n")
for hyp in hypo_cluster:
    print(hyp)

print("\n")



####### plot results
fig = plt.figure()
axis1 = fig.add_subplot(121)

# TODO - figure out how to distinguish cossim enough to be shown in radius of plots
# plt.scatter(X, y, s=cossim * np.pi * 100, c="b")
#axis1.scatter(wordX, wordy, s=5, c="k", label=sys.argv[0])

# TODO - figure out how to distinguish word from other synonyms
#axis1.scatter(synX[0], syny[0], s=20, c="k", label=sys.argv[1])              #attempt to do word in different style
#axis1.scatter(synX[1:], syny[1:], c="b", label="synonyms")
axis1.scatter(synX, syny, s=20, c="b", label="synonyms")
axis1.scatter(antX, anty, c="y", label="antonyms")
axis1.scatter(hyperX, hypery, c="r", label="hypernyms")
axis1.scatter(hypoX, hypoy, c="g", label="hyponyms")
#plot kmeans centroids
axis1.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="k", zorder=10)
#TODO - add KMeans decision boundaries

# create legend
plt.legend(loc="upper left")


#plot distance from centroids on second subplot
distX = np.array(map(lambda z: z[0], kmeans.transform(total_data)))
disty = np.array(map(lambda z: z[1], kmeans.transform(total_data)))

axis2 = plt.subplot(122)
axis2.scatter(distX, disty, c="k", label="distance from centroid")

#TODO add labels to second subplot


# label each point with its word and cosine similarity

#axis1.annotate(wordDict.keys[0], wordX[0], wordy[0])

#TODO = figure out error at line 243 (only in pca)
for i, word in enumerate(synDictKeys):
    axis1.annotate(word + ":" + str(round(synWorddict[word][1], 3)), (synX[i], syny[i]))

if antDictKeys:
    for i, word in enumerate(antDictKeys):
        axis1.annotate(word + ":" + str(round(antWorddict[word][1], 3)), (antX[i], anty[i]))

if hyperDictKeys:
    for i, word in enumerate(hyperDictKeys):
        axis1.annotate(word + ":" + str(round(hyperWorddict[word][1], 3)), (hyperX[i], hypery[i]))

if hypoDictKeys:
    for i, word in enumerate(hypoDictKeys):
        axis1.annotate(word + ":" + str(round(hypoWorddict[word][1], 3)), (hypoX[i], hypoy[i]))

plt.show()







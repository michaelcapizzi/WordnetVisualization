__author__ = 'mcapizzi'

from nltk import *
import gensim
import accessWordnet as nl
from sklearn import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import sys

#   Takes two arguments:
#       word to analyze
#       type of dimension reduction ("p" = PCA, "t" = TSNE)

# access word2Vec model
print("loading Word2Vec model")

model = gensim.models.word2vec.Word2Vec.load_word2vec_format("/home/mcapizzi/Google_Drive_Arizona/Programming/word2Vec/GoogleNews-vectors-negative300.bin.gz", binary=True)

print("building Wordnet relation sets")

# get word relations for word (imported from nltk_test)
synonyms = nl.get_wn_relation(sys.argv[1], "s")
antonyms = nl.get_wn_relation(sys.argv[1], "a")

# builds dict of word with [0] its word2vec vector, [1] cosine similarity to word, and [2] euclidean distance from word
wordDict = {}
synWorddict = {}
antWorddict = {}

# add word to worddict (cosine similarity and euclidean distance will be preset)
synWorddict[sys.argv[1]] = (model[sys.argv[1]], 1.0, 0.0)

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

synDictKeys = synWorddict.keys()
synDictValues = map(lambda z: synWorddict[z], synWorddict)

syn_all_data = np.array(map(lambda x: x[0], synDictValues))


if sys.argv[2] == "p":                                          # PCA - reduce to 2 dimensions
    syn_pca = decomposition.PCA(n_components=2)
    syn_output = np.array(syn_pca.fit_transform(syn_all_data))
elif sys.argv[2] == "t":
    syn_tsne = manifold.TSNE(n_components=2, random_state=0)    # t-SNE - reduce to 2 dimensions
    syn_output = np.array(syn_tsne.fit_transform(syn_all_data))
else:
    syn_pca = decomposition.PCA(n_components=2)
    syn_output = np.array(syn_pca.fit_transform(syn_all_data))


# get cartesian coordinates
synX = np.array(map(lambda z: z[0], syn_output))
syny = np.array(map(lambda z: z[1], syn_output))
syn_cossim = np.array(map(lambda z: z[2], synDictValues))

######antonyms

antDictKeys = antWorddict.keys()
antDictValues = map(lambda z: antWorddict[z], antWorddict)

ant_all_data = np.array(map(lambda x: x[0], antDictValues))


if sys.argv[2] == "p":                                          # PCA - reduce to 2 dimensions
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


####### KMeans

total_data = np.concatenate((syn_output, ant_output))

print("clustering Word2Vec vectors with k = 2")

#build 2 clusters, to test hypothesis of synonyms and antonyms clustered together
kmeans = KMeans(n_clusters=2)
kmeans.fit(total_data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

syn_cluster = kmeans.predict(syn_output)
#TODO add distribution count
(syn_cluster1, syn_cluster2) = (syn_cluster.tolist().count(0), syn_cluster.tolist().count(1))
print("\n")
print("synonyms are in the following clusters: " + str(syn_cluster).strip('[]'))
print("distribution: cluster #1 --> " + str(syn_cluster1) + ", cluster #2 --> " + str(syn_cluster2))
print("\n")

ant_cluster = kmeans.predict(ant_output)
#TODO add distribution count
(ant_cluster1, ant_cluster2) = (ant_cluster.tolist().count(0), ant_cluster.tolist().count(1))
print("\n")
print("antonyms are in the following clusters: " + str(ant_cluster).strip('[]'))
print("distribution: cluster #1 --> " + str(ant_cluster1) + ", cluster #2 --> " + str(ant_cluster2))
print("\n")

####### plot results
fig = plt.figure()
axis1 = fig.add_subplot(111)

#plot all Word2Vec vectors with dimensions reduced
if synX.any():
    axis1.scatter(synX, syny, s=20, c="b", label="synonyms")
if antX.any():
    axis1.scatter(antX, anty, c="y", label="antonyms")

#plot kmeans centroids
axis1.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="k", zorder=10, label="cluster centroids")
plt.title("Word2Vec vectors in reduced dimensionality")
# create legend
plt.legend(loc="upper left")

# label each point with its word and cosine similarity
for i, word in enumerate(synDictKeys):
    axis1.annotate(word + ":" + str(round(synWorddict[word][1], 3)), (synX[i], syny[i]))

if antDictKeys:
    for i, word in enumerate(antDictKeys):
        axis1.annotate(word + ":" + str(round(antWorddict[word][1], 3)), (antX[i], anty[i]))

plt.show()







from umap import UMAP as UMAP_raw
from sklearn.decomposition import PCA as PCA_raw
from sklearn.cluster import KMeans as KMeans_raw
from sklearn.manifold import TSNE as TSNE_raw

# Force random_state to be a fixed int to stabilize algorithms
# for debugging
ADDL_OPTS = {"random_state": 42}
#ADDL_OPTS = {}

UMAP_ADDL_OPTS = ADDL_OPTS
PCA_ADDL_OPTS = ADDL_OPTS
KMEANS_ADDL_OPTS = ADDL_OPTS

def UMAP(*argv, **argc):
    my_argc = UMAP_ADDL_OPTS.copy()
    my_argc.update(argc)
    return UMAP_raw(*argv, **my_argc)

def PCA(*argv, **argc):
    my_argc = PCA_ADDL_OPTS.copy()
    my_argc.update(argc)
    return PCA_raw(*argv, **my_argc)

def KMeans(*argv, **argc):
    my_argc = KMEANS_ADDL_OPTS.copy()
    my_argc.update(argc)
    return KMeans_raw(*argv, **my_argc)

def TSNE(*argv, **argc):
    my_argc = KMEANS_ADDL_OPTS.copy()
    my_argc.update(argc)
    return TSNE_raw(*argv, **my_argc)

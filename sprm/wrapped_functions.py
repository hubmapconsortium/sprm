import logging

from frozendict import frozendict

# from sklearn.cluster import KMeans as KMeans_raw
from sklearn.cluster import MiniBatchKMeans as KMeans_raw
from sklearn.decomposition import PCA as PCA_raw
from sklearn.manifold import TSNE as TSNE_raw
from umap import UMAP as UMAP_raw

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

#
# Uncomment the following to set random seeds. This
# helps to reduce run-to-run variability.
#
# ADDL_OPTS = frozendict({"random_state": 42})
ADDL_OPTS = frozendict()

UMAP_ADDL_OPTS = ADDL_OPTS
PCA_ADDL_OPTS = ADDL_OPTS
KMEANS_ADDL_OPTS = ADDL_OPTS


def UMAP(*argv, **argc):
    my_argc = dict(UMAP_ADDL_OPTS)
    my_argc.update(argc)
    return UMAP_raw(*argv, **my_argc)


def PCA(*argv, **argc):
    my_argc = dict(PCA_ADDL_OPTS)
    my_argc.update(argc)
    return PCA_raw(*argv, **my_argc)


class KMeans(KMeans_raw):
    def __init__(self, **argc):
        my_argc = dict(KMEANS_ADDL_OPTS)
        my_argc.update(argc)
        n_clusters = my_argc.get("n_clusters", 8)
        init = my_argc.get("init", "k-means++")
        n_init = my_argc.get("n_init", "auto")
        max_iter = my_argc.get("max_iter", 300)
        tol = my_argc.get("tol", 0.0001)
        verbose = my_argc.get("verbose", 0)
        random_state = my_argc.get("random_state", None)
        copy_x = my_argc.get("copy_x", True)
        algorithm = my_argc.get("algorithm", "lloyd")
        for key in my_argc:
            if key not in frozenset(
                [
                    "n_clusters",
                    "init",
                    "n_init",
                    "max_iter",
                    "tol",
                    "verbose",
                    "random_state",
                    "copy_x",
                    "algorithm",
                ]
            ):
                raise RuntimeError(f"Unknown key {key}")
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            # copy_x=copy_x,
            # algorithm=algorithm
        )
        LOGGER.debug(f"Kmeans init {my_argc}")

    def fit(self, x, *argv, **argc):
        LOGGER.debug(f"KMeans fit begin {x.shape} {x.dtype}")
        rslt = super().fit(x, *argv, **argc)
        LOGGER.debug("KMeans fit end")
        return rslt


def TSNE(*argv, **argc):
    my_argc = dict(KMEANS_ADDL_OPTS)
    my_argc.update(argc)
    return TSNE_raw(*argv, **my_argc)

import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS
import HyperboloidMDS as hm
from HyperboloidMDS import CliqueDecompositionEmbedding
import time


def buildKNNGraph(deg,distMat,fitDist):
    G = nx.Graph()
    N=distMat.shape[0]
    distDict = {}
    fitDistDict = {}
    for i in range(N):
        G.add_node(i)
    for i in range(N):
        index = np.argsort(distMat[i, :])
        for j in range(1, deg + 1):
            G.add_edge(i, index[j], distance=distMat[i, index[j]])
    DELvect = np.zeros(G.number_of_edges())
    ind = 0
    for u, v, data in G.edges(data=True):
        distDict[u] = {v: data['distance']}
        distDict[v] = {u: data['distance']}
        DELvect[ind] = data['distance']
        fitDistDict[u] = {v: dist}
        fitDistDict[v] = {u: dist}
        ind += 1
    for node in self.G.nodes():
        del G.nodes[node]['X']
    return [delVect,G]



if __name__ == "__main__":
    path = "./GeoHyp_Embedding"
    filename="usca312.edges"
    usca312=np.loadtxt(path+filename)
    distMat=np.zeros((312,312))
    for i in range(usca312.shape[0]):
        distMat[int(usca312[i][0]),int(usca312[i][1])]=usca312[i][2]
    N=312
    sum=0
    dim=50
    C=1
    fitDist=np.zeros(312)
    fitDist = np.cosh(distMat)
    mat = np.ones((N, N)) * fitDist[0, :]
    fitDist=2*fitDist + (mat-mat.transpose())**2-2
    #validation 2: Check eigenvalues
    mat=np.ones((N, N)) * fitDist[0, :]
    D=-1/2*(fitDist-mat-mat.transpose())
    #check how many eigenValues are >0
    #check what is the rank of the EMD matrix
    [l,U]=np.linalg.eigh(D)
    l=np.real(l)
    U=np.real(U)
    print("Eigenvalues:",l[0:dim+5])
    print("rank:",np.linalg.matrix_rank(fitDist))

    #Validation 3: check reconstruction
    Xfit=np.zeros((N,dim))
    fit=np.matmul(np.diag(np.sqrt(l[-dim+1:])),U.transpose()[-dim+1:,:])
    Xfit[:,0:-1]=fit.transpose()
    Xfit[:,-1]=np.sqrt(np.sum(Xfit[:,0:-1]**2,axis=1)+1)
    estDistMat = np.zeros((N, N))
    for i in range(N):
        X=hm.Hyperpoint(Xfit[i, :],C)
        for j in range(i + 1, N):
            Y=hm.Hyperpoint(Xfit[j, :],C)
            estDistMat[i, j] = X.hdist(Y)
            estDistMat[j, i] = estDistMat[i,j]
    sum=np.sum((estDistMat-distMat)**2)
    print("Distance Error:", sum)

    [delVect,G]=buildKNNGraph(5, distMat)



####################################################



    path = "./"
    filename="mammal_closure.graphml"
    G=nx.read_graphml(path+filename)
    G = nx.convert_node_labels_to_integers(G)
#    for u,v,data in G.edges(data=True):
#        data['ricciDistance']=data['distance']
#        data['distance']=1

    GG=G.copy()
    verticesSet=list(GG.nodes())
    hyper=hm.HyperHypNew(verticesSet,GG,1,10,1)
    print("Hyperboloid gradient")
    start_time = time.time()
    coords=hyper.HYP_fit()
    print("Execution time: ", time.time()-start_time)
    utils=hm.Utils()
#    print("Hyperboloid MAP:", utils.MAP(GG,coords))
    print("Hyperboloid gradient rmse:", hyper.rmse)
    del hyper

    GG=G.copy()
    verticesSet=list(GG.nodes())
    hyper=hm.HyperMDS(verticesSet, GG, 1, 10, 1, )
    print("Euclidean gradient")
    start_time = time.time()
    hyper.MDS_fit()
    print("Execution time: ", time.time() - start_time)
    print("Euclidean Gradient rmse:", hyper.rmse)
    utils=hm.Utils()
    print("Euclidean MAP:", utils.MAP(GG,coords))
    del hyper

    GG=G.copy()
    print("clique decomposition")
    start_time = time.time()
    hyper=hm.HypMDSEmbedding(GG,1)
    print("Clique decomposition creation time: ", time.time() - start_time)
    start_time = time.time()
    hyper.fit(3)
    print("Execution time: ", time.time() - start_time)
    print("Clique decomposition rmse:", hyper.rmse)
    utils=hm.Utils()
    print("Clique decomposition MAP:", utils.MAP(GG,coords))
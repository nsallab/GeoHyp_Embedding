import networkx as nx
from numpy.core.defchararray import zfill
#from threadpoolctl import threadpool_limits
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy import optimize
from tqdm import tqdm
#import mpmath as mp
import random
import matplotlib.pyplot as plt

#import mpmath as mp

#from pipeline import adjMat

#threadpool_limits(limits=15, user_api='blas')
from scipy.sparse import triu, csr_matrix

class Utils:
    def __init__(self,G=None):
        self.G=G
    def MAP(self,G,X):
        def localDist(u,X):
            x=X[u,:]
            a=x[0]*X[:,0]+x[1]*X[:,1]-x[2]*X[:,2]
            a[a>-1]=-1
            D=np.arccosh(-a)
            DD=np.sort(D)
            I=np.argsort(D).tolist()
            I.remove(u)
            return(I)

        MAP=0.0
        for u in G.nodes():
            N=list(G.neighbors(u))
            I=localDist(u,X)
            val=0.0
            for v in N:
                if u != v:
                    ind=I.index(v)
                    R=I[0:ind + 1]
                    inter=set(N) & set(I[0:ind + 1])
                    val+=len(inter)/len(R)
                MAP +=val/G.degree[u]
        return MAP/nx.number_of_nodes(G)

    def plot(X,Y):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x=X, y=Y)
        plt.xlabel("Target Distance")
        plt.ylabel("error after")
        plt.show()

    def distorsion(G,C):
        D=np.zeros(G.number_of_edges())
        estD=np.zeros(G.number_of_edges())
        inc=0
        for u,v, data in G.edges(data=True):
            X=Hyperpoint(np.array(G.nodes[u]['Coords']),C)
            Y=Hyperpoint(np.array(G.nodes[v]['Coords']),C)
            D[inc]=data['distance']
            estD[inc]=X.hdist(Y)
            inc+=1
        sse=np.sum((D-estD)**2)
        rmse=np.sqrt(sse/G.number_of_edges())
        srse=np.sum((1-estD/D)**2)
        rrmse=np.sqrt(srse/G.number_of_edges())
        return (sse,rmse, srse,rrmse,D, estD)
    def toPD(self,X):
        return(X[:,0]/(X[:,2]+1)+1j*X[:,1]/(X[:,2]+1))

    def toPP(self,X): #to Poincaré Plan
        return np.vstack((X[:,0]/(X[:,2]-X[:,1]),1/(X[:,2]-X[:,1]))).transpose()

    def toPS(self,X):
        A=np.sqrt(X[:,0]**2-1)/(X[:,0])
        x=A-np.arctanh(A)
        y=np.cos(X[:,1])/X[:,0]
        z=np.sin(X[:,1])/X[:,0]

    def hypMoveCenter(self,X, X0):
        v = np.arccosh(X0[2])
        if X0[1] > 0:
            theta = np.arccos(X0[0] / np.sinh(v))
        else:
            theta = -np.arccos(X0[0] / np.sin(v))
        rTheta = np.zeros((3, 3))
        rTheta[0, 0] = np.cos(-theta)
        rTheta[0, 1] = -np.sin(-theta)
        rTheta[1, 0] = np.sin(-theta)
        rTheta[1, 1] = np.cos(-theta)
        rTheta[2, 2] = 1
        sV = np.zeros((3, 3))
        sV[0, 0] = np.cosh(-v)
        sV[0, 2] = np.sinh(-v)
        sV[2, 0] = np.sinh(-v)
        sV[2, 2] = np.cosh(-v)
        sV[1, 1] = 1
        trans = np.matmul(sV, rTheta)
        return (np.matmul(trans, X))


class Hyperpoint:
    def __init__(self,X0, C):
        self.X=X0
        self.C=C
        self.K=1/np.sqrt(C)
        self.dim=self.X.shape[0]-1

    def B(self,Y):
        a=np.sum(self.X[0:-1] * Y.X[0:-1]) - self.X[-1] * Y.X[-1]
        if a>-1:
            a=-1
        return a

    def Q(self):
        return self.B(self.X,self.X)

    def hdist(self,Y):
        return (self.K**2)*np.arccosh(-self.B(Y))

class MetricGraph:
    def __init__(self,G):
        self.G=G
        self.D=np.zeros(G.number_of_edges())
        ind =0
        self.fitDist=np.full(G.number_of_edges(),np.nan)
        for u,v, data in G.edges(data=True):
            self.D[ind]=data['distance']
            ind +=1
        self.distMat=np.array(0)

    def setdistFit(self, K, origin):
        dist0=self.getdist0(origin)
        ind =0
        for u,v, data in self.G.edges(data=True):
            self.D[ind]=data['distance']
            self.fitDist[ind]=2*np.cosh(self.D[ind]/K**2)-(np.cosh(dist0[u]/K**2)-np.cosh(dist0[v]/K**2))**2-2
            ind +=1
        return(self.fitDist)

    def getdist0(self, origin):
        if self.distMat.shape ==() :
            dist0=np.zeros(self.G.number_of_nodes())
            distDict = (nx.shortest_path_length(self.G, source=origin, weight='distance'))
            for n in distDict:
                dist0[n]=distDict[n]
        else:
            dist0 = self.distMat[origin, :]
        return dist0

class SyntheticPointCloud:
    def __init__(self, N, dim, C):
        self.N = N
        self.dim = dim
        self.C = C
        self.K = 1/np.sqrt(C)
        self.X = np.zeros((N, dim+1))
        self.X[:,0:-1] = np.random.rand(N, dim)
        self.X[:,-1] = np.sqrt(np.sum(self.X[:,:-1]**2, axis=1)+1)
        self.X[0,:]=np.zeros(dim+1)
        self.X[0,-1]=1
        self.distMat = np.zeros((N, N))
        for i in range(N):
            X=Hyperpoint(self.X[i,:],self.C)
            for j in range(i+1,N):
                Y = Hyperpoint(self.X[j, :], self.C)
                self.distMat[i,j] = X.hdist(Y)
                self.distMat[j,i] = self.distMat[i,j]
        mat = np.ones((self.N,self.N))*np.cosh(self.distMat[0,:]/self.K**2)
        mat = mat-mat.transpose()
        self.fitDist = np.zeros((self.N,self.N))
        self.fitDist =  2*np.cosh(self.distMat/self.K**2)+mat**2-2

    def rebuild(self,C):
        self.C=C
        ind=0
        self.distMat = np.zeros((self.N, self.N))
        for i in range(self.N):
            X=Hyperpoint(self.X[i,:],self.C)
            for j in range(i+1,self.N):
                Y = Hyperpoint(self.X[j, :], self.C)
                self.distMat[i,j] = X.hdist(Y)
                self.distMat[j,i] = self.distMat[i,j]
        mat = np.ones((self.N,self.N))*np.cosh(self.distMat[0,:]/self.K**2)
        mat = mat-mat.transpose()
        self.fitDist = np.zeros((self.N,self.N))
        self.fitDist =  2*np.cosh(self.distMat/self.K**2)+mat**2-2

    def buildKNNGraph(self, deg):
        G = nx.Graph()
        self.distDict={}
        self.fitDistDict={}
        for i in range(self.N):
            G.add_node(i,X=self.X[i,:])
        for i in range(self.N):
            index=np.argsort(self.distMat[i,:])
            for j in range(1,deg+1):
                G.add_edge(i, index[j], distance=self.distMat[i,index[j]])
        self.MG=MetricGraph(G)
        dist0=self.distMat[0,:]
        self.MG.setdistFit(self.K,0)
        return self.MGG

    def buildRandomGraph(self, deg):
        G = nx.Graph()
        for i in range(self.N):
            G.add_node(i,X=self.X[i,:])
        nodes = list(G.nodes())
        for i in range(self.N):
            for j in range(deg):
                j=random.choice(nodes)
                if (i!=j):
                    X1 = Hyperpoint(G.nodes[i]['X'], self.C)
                    X2 = Hyperpoint(G.nodes[j]['X'], self.C)
                    G.add_edge(i, j, distance=X1.hdist(X2))
        self.MG=MetricGraph(G)
        self.MG.distMat=self.distMat
        self.MG.setdistFit(self.K, 0)
        return self.MG

    def buildFullGraph(self):
        G=nx.Graph()
        for i in range(self.N):
            G.add_node(i,X=self.X[i,:])
        for i in range(self.N):
            for j in range(i+1,self.N):
                self.G.add_edge(i,j,distance=self.distMat[i,j])
        self.MG=MetricGraph(G)
        dist0=self.distMat[0,:]
        self.MG.setdistFit(self.K,0)
        return self.MGG


class CliqueDecomposition:
    def __init__(self, G, C):
        self.G = G
        self.dcount=0
        self.anchored=set()
        self.K=1/np.sqrt(C)

    def complete_to_chordal1(self):
        self.H = self.G.copy()
        alpha = {node: 0 for node in self.H}
        if nx.is_chordal(self.H):
            return  self.H
        chords = set()
        weight = {node: 0 for node in self.H.nodes()}
        unnumbered_nodes = list(self.H.nodes())
        for i in tqdm(range(len(self.H.nodes()), 0, -1)):
            # get the node in unnumbered_nodes with the maximum weight
            z = max(unnumbered_nodes, key=lambda node: weight[node])
            unnumbered_nodes.remove(z)
            alpha[z] = i
            update_nodes = []
            for y in unnumbered_nodes:
                if self.G.has_edge(y, z):
                    update_nodes.append(y)
                else:
                    # y_weight will be bigger than node weights between y and z
                    y_weight = weight[y]
                    lower_nodes = [
                        node for node in unnumbered_nodes if weight[node] < y_weight
                    ]
                    if nx.has_path(self.H.subgraph(lower_nodes + [z, y]), y, z):
                        update_nodes.append(y)
                        chords.add((z, y))
            # during calculation of paths the weights should not be updated
            for node in update_nodes:
                weight[node] += 1
        self.H.add_edges_from(chords)
        self.alpha = alpha
        return self.H, alpha


    def getCliqueStats(self,cliqueSet):
        subgraph=self.G.subgraph(cliqueSet)
        if subgraph.number_of_edges()==0:
            print("no cliques found")
        dists=np.zeros(subgraph.number_of_edges())
        i=0
        for u,v,data in subgraph.edges(data=True):
            dists[i]=data["distance"]
            i +=1
        return np.mean(dists), np.std(dists)


    def checkClique(self, Set):
        subgraph=self.G.subgraph(Set)
        M = max(subgraph.degree, key=lambda x:x[1])
        m = min(subgraph.degree, key=lambda x:x[1])
        if M[1] !=m[1] or M[1]!=len(Set)-1:
            return False
        else:
            return True

    def buildCliqueTree(self):
        for u in self.G.nodes():
            self.G.nodes[u]["Cv"] = set()
            for v in self.G.neighbors(u):
                if self.alpha[v] > self.alpha[u]:
                    self.G.nodes[u]["Cv"].add(v)
        maxLen = 0
        maxClique = set()
        self.mycliqueTree = None
        cliqueTree = self.mycliqueTree
        currentGraph = self.G.copy()
        sorted_nodes = sorted(self.alpha.items(), key=lambda kv: (kv[1], kv[0]))

        keys=list(self.alpha.keys())
        for i in tqdm(range(0, len(self.alpha))):
            u=sorted_nodes[i][0]
#            u = 'n'+str(self.alpha[keys[i]] - 1)
            if u in currentGraph:
                verticesSet = set(currentGraph.nodes())
                if True:
#                if self.G.nodes[u]["Cv"] and (not self.checkClique(self.G.nodes[u]["Cv"])):
#                    print("Not clique", u, self.G.nodes[u]["Cv"])
#                else:
                    for subgraph in nx.connected_components(self.G.subgraph(verticesSet - self.G.nodes[u]["Cv"])):
                        if u in subgraph:
                            A = subgraph
                            break
                    C = currentGraph.nodes[u]["Cv"]
                    if len(C)>0:
                        left = A.union(C)
                        B = verticesSet - (left)
                        avgC,stdC = self.getCliqueStats(C)

#                        if B and avgC<0.1:
                        if True:
                            if cliqueTree is None:
                                cliqueTree = Tree()
                                self.mycliqueTree = cliqueTree
                            else:
                                cliqueTree.right=Tree()
                                cliqueTree=cliqueTree.right
                            right = B.union(C)
                            cliqueTree.data = C
                            cliqueTree.left = left
                            cliqueTree.right = right
                            cliqueTree.stats=(self.getCliqueStats(right),self.getCliqueStats(left),self.getCliqueStats(C))
                            currentGraph = currentGraph.subgraph(right)
#                            print("left Atom:",len(left),self.getCliqueStats(left))
                            print("left Atom:", len(left))
#                            print("right sub Graph:",len(right),self.getCliqueStats(right))
                            print("right sub Graph:", len(right))
#                            print("Clique :", len(C), self.getCliqueStats(C))
                            print("Clique :", len(C))


                L = len(self.G.nodes[u]["Cv"])
                if L > maxLen:
                    maxClique = self.G.nodes[u]["Cv"]
                    maxLen = L
        self.maxCliqueDim=maxLen+1
        return self.mycliqueTree

    def tree2List(self):
        ctree=self.mycliqueTree
        l=[]
        while isinstance(ctree,Tree):
           l.append([ctree.left,ctree.data, ctree.stats])
           ctree=ctree.right
        l.append([ctree])
        return l

    def list2Tree(self,l):
        self.mycliqueTree=Tree()
        tree=self.mycliqueTree
        pastTree=None
        for e in l[:-1]:
            tree.right=Tree()
            tree.left=e[0]
            tree.data=e[1]
            tree.stats=e[2]
            tree.father=pastTree
            pastTree=tree
            tree=tree.right
        pastTree.right=l[-1][0]


class Tree:
    def __init__(self):
        self.right = None
        self.left = None
        self.data = None
        self.father = None

class Simplex:
    def __init__(self, anchors):
        self.anchors = anchors
    def check(self, u,G):
        anchors=[]
        for anchor in self.anchors:
            if G.has_edge(u, anchor):
                anchors.append(anchor)
        return anchors

    def add(self, anchor):
        self.anchors.append(anchor)

    def getDim(self):
        return(len(self.anchors))

class HypMDSEmbedding:
    def __init__(self,MG,C, tree=None,coords=None):
        self.MetricGraph=MG
        self.G=MG.G
        self.C=C
        self.D0=np.zeros(self.G.number_of_edges())
        i=0
        for u,v,data in self.G.edges(data=True):
            self.D0[i]=data['distance']
            i+=1
        if tree is None:
            cliqueDecomposition = CliqueDecomposition(self.G, 1)
            cliqueDecomposition.complete_to_chordal1()
            cliqueDecomposition.buildCliqueTree()
            self.tree= cliqueDecomposition.mycliqueTree
        else:
            self.tree = tree
        self.coords = coords

    def checkTree(self):
        tree=self.tree
        nodes=set(self.G.nodes())
        returnTree = None
        while isinstance(tree,Tree):
            verticesSet = tree.left
            nodes -=verticesSet
            nodes -= tree.data
            tree=tree.right
        if nodes:
            print("problem")


    def fit(self,dim, distMat=None):
        tree=self.tree
        matched=set()
        returnTree=None
        Match=False
        G=self.MetricGraph.G
        while isinstance(tree,Tree):
            verticesSet = tree.left
            if len(matched)==0 or len(matched&set(verticesSet))>0:
                if len(matched&set(verticesSet))<len(verticesSet):
                    Match=True
                    for s in nx.connected_components(G.subgraph(verticesSet)):
                        hyper = HyperHyp(s, self.MetricGraph, matched,self.C, dim, stress='rmse')
                        hyper.HYP_fit()
                        matched.update(s)
                        print("Atom size:", len(s))
                        print("RMSE Error:", hyper.rmse)
                        del hyper
                    if returnTree:
                        tree = returnTree
                        returnTree=None
                        continue
            else:
                if not returnTree:
                    returnTree=tree
            tree=tree.right
        verticesSet=tree
        hyper = HyperMDS(verticesSet, self.MetricGraph, self.C, dim, 1, verb=1, stress='rmse')
        hyper.MDS_fit(self.coords)
        print("Atom size:", len(verticesSet))
        print("RMSE Error:",hyper.rmse)
        del hyper
        self.DELvect=np.zeros(G.number_of_edges())
        self.D=np.zeros(G.number_of_edges())
        cnt=0
        for u,v,data in G.edges(data=True):
            XX=Hyperpoint(G.nodes[u]["Coords"],self.C)
            YY=Hyperpoint(G.nodes[v]["Coords"],self.C)
            self.D[cnt]=XX.hdist(YY)
            self.DELvect[cnt]=data['distance']
            cnt +=1
        self.sserror=np.sum((self.DELvect-self.D)**2)
        self.rmse=np.sqrt(self.sserror/G.number_of_edges())


#class for hyperboloid gradient method
class HyperHyp:
    def __init__(self,nodeSet, fullMetricGraph, anchored, C, dim,  verb=1, origin=None, dist0=None, stress='rmse'):
        self.verb = verb
        self.anchored = anchored
        self.MetricGraph = fullMetricGraph
        self.fullGraph = fullMetricGraph.G
        self.G=nx.subgraph(self.fullGraph,nodeSet).copy()
        self.num=self.G.number_of_nodes()
        self.C=C
        self.K=1/np.sqrt(C)
        self.dim=dim
        self.anchors=[]
        self.anchorsCoords=[]
        self.nonAnchorMask=np.full(self.num*self.dim,True)
        self.anchorMask=np.full(self.num*self.dim,False)
        self.anchorEdge=[]
        self.nonAnchorEdge=[]
        self.X0=np.zeros(self.dim*self.num)
        self.nodes=[]
        self.edges=[]
        self.D0 = np.zeros(self.G.number_of_edges())
        self.DELvect = np.zeros(self.G.number_of_edges())
        self.fitDist = np.zeros(self.G.number_of_edges())
        self.translate= False
        i=0
        for u,v,data in self.G.edges(data=True):
            self.D0[i]=data['distance']
            i+=1
        for n in self.G.nodes():
            self.nodes.append(n)
            if n in self.anchored:
                self.anchors.append(n)
                self.anchorsCoords.append(self.G.nodes[n]['Coords'])
        ind = 0
        self.incMat_s=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.incMat_t=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.dist0=np.zeros(self.G.number_of_nodes())
        self.sources=[]
        self.targets=[]
        oorigin = np.zeros(self.dim + 1)
        oorigin[-1] = 1
        if self.anchors :
            if len(self.anchors) < self.G.number_of_nodes():
                #we need to fit
                self.toFit=True
                anchorsCoords = np.array(self.anchorsCoords)
                if origin is None:
                    self.origin = self.anchors[0]
                else:
                    if origin in self.anchors:
                        self.origin = origin
                    else:
                        self.origin=self.anchors[0]
                if not (np.array_equal(origin, self.anchorsCoords[0])):
                    #we need to translate the choosen vertex to the origin
                    self.translate = True
                    self.PP = anchorsCoords[0]
                    anchorsCoords = self.translationToOrigin(anchorsCoords, np.array(anchorsCoords[0]))
#                    ind +=1
                    for anchor in self.anchors:
                        I = self.nodes.index(anchor)
                        self.anchorMask[I::self.num] = True
                        self.nonAnchorMask[I::self.num] = False
                        self.X0[I::self.num] = anchorsCoords[ind,0:-1]
                        ind +=1
            else:
                # all vertices are already anchored
                # we have nothing to do
                ind = 0
                for u, v, data in self.G.edges(data=True):
                    self.edges.append((u, v))
                    self.DELvect[ind] = data['distance']
                    I = self.nodes.index(u)
                    J = self.nodes.index(v)
                    self.sources.append(I)
                    self.targets.append(J)
                    self.incMat_s[I, ind] = 1
                    self.incMat_t[J, ind] = 1
                    if (u in self.anchors) and (v in self.anchors):
                        self.anchorEdge.append(ind)
                    else:
                        self.nonAnchorEdge.append(ind)
                    ind +=1
                self.toFit=False
        else:
            self.toFit=True
            if oorigin is None:
                self.origin = list(self.G.nodes)[0]
            else:
                if origin is None:
                    self.origin = list(self.G.nodes)[0]
                else:
                    self.origin = origin
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1

            self.fullGraph.nodes[self.origin]['Coords'] = oorigin
            self.G.nodes[self.origin]['Coords'] = oorigin
            self.anchorMask[0::self.num] = True
            self.nonAnchorMask[0::self.num] = False
            self.X0[0::self.num] = oorigin[0:-1]
            self.anchors.append(self.origin)
            self.translate = False
        if self.toFit:
            ind=0
            for u, v, data in self.G.edges(data=True):
                self.edges.append((u, v))
                self.DELvect[ind] = data['distance']
                I = self.nodes.index(u)
                J = self.nodes.index(v)
                self.sources.append(I)
                self.targets.append(J)
                self.incMat_s[I, ind] = 1
                self.incMat_t[J, ind] = 1
                if (u in self.anchors) and (v in self.anchors):
                    self.anchorEdge.append(ind)
                else:
                    self.nonAnchorEdge.append(ind)
                ind += 1

        #        if stress == 'rmse':
#            self.fgrad = self.fgrad_rmse
#            self.fstress = self.fstress_rmse
#        elif stress == 'rrmse':
#            self.fgrad = self.fgrad_rrmse
#            self.fstress=self.fstress_rrmse
#        elif stress == 'sam':
#            self.fgrad = self.fgrad_sam
#            self.fstress=self.fstress_sam
#        self.translate = False

    def hdistmat(self, Xsources, Xtargets,K=None):
        #Xsources is a matrix with n rows and k+1 columns
        #Xtargets is a matrix with n rows and k+1 columns
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        if K is None:
            K=self.K
        if np.where(B>-1):
            B[np.where(B>-1)]=-1
        D=(K**2)*np.arccosh(-B)
        return D

    def Bmat(self,Xsources, Xtargets):
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        return B

    def translationToOrigin(self,X,PP):
    # Calculate the new positions of X when origin move to position P
        P=PP.copy()
        P[0:-1]=-P[0:-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.matmul(mat, X.transpose())
        return translated.transpose()

    def translationBack(self,X,PP):
        P=PP.copy()
        P[-1]=P[-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.dot(mat, X.transpose())
        return translated.transpose()

    def HYP_cost(self,X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1]**2, axis=1)+1)
        Xsources=X[sources][self.nonAnchorEdge]
        Xtargets=X[targets][self.nonAnchorEdge]
        D=self.hdistmat(Xsources, Xtargets)
        dd=d[self.nonAnchorEdge]
        F = np.sum((D- dd)** 2)
        return F

    def HYP_gradient(self, X1, sources, targets, d):
        # X is a vector of nk values
        # XX is a matrix of k points in k+1 dimension and an array of k values
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        K=self.K
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1]**2, axis=1)+1)
        Xsources=X[sources,:][self.nonAnchorEdge]
        Xtargets=X[targets,:][self.nonAnchorEdge]
        D=self.hdistmat(Xsources, Xtargets,K)
        dd=d[self.nonAnchorEdge]
        err=D-dd
        B=self.Bmat(Xsources, Xtargets)
        if np.where(B>-1)[0].size>0:
            print("problem")
        dE=np.zeros((num,self.dim))
        coupling_s = Xtargets[:,0:-1]-((Xtargets[:,-1]/Xsources[:,-1])*Xsources[:,0:-1].T).T
        coupling_t = Xsources[:,0:-1]-((Xsources[:,-1]/Xtargets[:,-1])*Xtargets[:,0:-1].T).T
        coef=-2*K**2*err/np.sqrt(np.maximum(B**2-1,1e-16))
        val_s=(coef*coupling_s.T).T
        val_t=(coef*coupling_t.T).T
        for k in range(self.dim):
            dE[:,k]=self.incMat_s[:,self.nonAnchorEdge]@val_s[:,k]+self.incMat_t[:,self.nonAnchorEdge]@val_t[:,k]
        dEE=dE.T.reshape(-1).ravel()[self.nonAnchorMask]
        return dEE

    def HYP_jacobian(self, X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0, :] = np.zeros((1, self.dim))
        Xsources=X[sources]
        Xtargets=X[targets]
        dX=Xtargets-Xsources
        F = (np.sum(dX ** 2, axis=1) - d ** 2)
        EE = np.zeros((num*self.dim, num*self.dim))
        for k in range(self.dim):
            FF = np.zeros((len(sources), num*self.dim))
            for l in range(self.dim):
                for m in range(len(sources)):
                    if l == k:
                        FF[m, sources[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                        FF[m, targets[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                    else:
                        FF[m, sources[m] + l * num]=2*dX[m,k]*dX[m,l]
                        FF[m, targets[m] + l * num]=2*dX[m,k]*dX[m,l]
            E = self.incMat.dot(FF)
            EE[k*num:(k+1)*num, :] = E
#        EE[0,:].fill(0)
#        EE[:,0].fill(0)
        return 4*EE[np.ix_(self.nonAnchorMask,self.nonAnchorMask)]



    def HYP_fit(self,X0=None):
#        np.random.seed(0)
        if self.toFit :
            numVar=np.sum(self.nonAnchorMask)//self.dim
            if X0 is None:
                numNode=self.nonAnchorMask.size//self.dim
                X0 = np.random.rand(numNode, self.dim)
            XX0=X0.T.reshape(-1)[self.nonAnchorMask]
            roots = minimize(self.HYP_cost, XX0.reshape((1, numVar*self.dim), order='F').ravel(),
                            args=(self.sources, self.targets, self.DELvect), method='L-BFGS-B', jac=self.HYP_gradient,#hess=self.MDS_jacobian,
                             options={'disp': True, 'maxiter': 1000, 'maxiter':5000})
           #hess = self.jacobian
    #        roots = minimize(self.cost, X0.reshape((1, self.G.number_of_nodes() * self.dim), order='F').ravel(),
    #                         args=(self.sources, self.targets, self.DELvect), method='Nelder-Mead', jac=self.equations
    #                         , options={'disp': True, 'maxiter': 1000, 'verbose': 3})
            XX = roots['x']
            X = self.X0.copy()
            X[self.nonAnchorMask] = XX
            X = X.reshape((self.G.number_of_nodes(), self.dim), order='F')
            self.extendtoHyperboloid(X)

        else:
            self.coords=np.array(self.anchorsCoords)
        Xsources = self.coords[self.sources, :][self.nonAnchorEdge]
        Xtargets = self.coords[self.targets, :][self.nonAnchorEdge]
        self.D=self.hdistmat(Xsources, Xtargets)
        self.delta=self.D-self.DELvect[self.nonAnchorEdge]
        self.sserror=np.sum(self.delta**2)
        if self.G.number_of_edges() > 0:
            self.rmse=np.sqrt(self.sserror/self.G.number_of_edges())
        else:
            self.rmse=0.0
        if self.rmse != self.rmse :
            pass

        self.rerror=np.sum((self.delta/self.DELvect[self.nonAnchorEdge])**2)
        for u,v,data in self.G.edges(data=True):
            self.G[u][v]['edist'] = \
            self.hdistmat(np.array(self.G.nodes[u]['Coords'])[:, np.newaxis].transpose(),
                  np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                  self.K)[0]
            self.fullGraph[u][v]['edist'] = self.G[u][v]['edist']
        return self.coords,self.K

    def embed_graph(self,G=None,tree=None):
        if G is None:
            G=self.fullGraph

    def extendtoHyperboloid(self,X0):
        self.coords=np.zeros((X0.shape[0],self.dim+1))
        self.coords[:,0:-1]=X0
        self.coords[:,-1]=np.sqrt(np.sum(X0**2, axis=1)+1)
        if self.translate:
            self.coords=self.translationBack(self.coords,self.PP)
        for i in range(X0.shape[0]):
            self.fullGraph.nodes[self.nodes[i]]['Coords']=self.coords[i]
            self.fullGraph.nodes[self.nodes[i]]['anchored']=1
            self.G.nodes[self.nodes[i]]['Coords'] = self.coords[i]
            self.G.nodes[self.nodes[i]]['anchored']=1

        for u,v,data in self.G.edges(data=True):
            X1=Hyperpoint(self.G.nodes[u]['Coords'],self.C)
            X2=Hyperpoint(self.G.nodes[v]['Coords'],self.C)
            self.fullGraph[u][v]['estDist']=X1.hdist(X2)
            data['estDist']=X1.hdist(X2)



#class for hyperboloid gradient method
class HyperPD:
    def __init__(self,nodeSet, fullMetricGraph, anchored, C, dim,  verb, origin=None, dist0=None, stress='rmse'):
        self.verb = verb
        self.anchored = anchored
        self.MetricGraph = fullMetricGraph
        self.fullGraph = fullMetricGraph.G
        self.G=nx.subgraph(self.fullGraph,nodeSet).copy()
        self.num=self.G.number_of_nodes()
        self.C=C
        self.K=1/np.sqrt(C)
        self.dim=dim
        self.anchors=[]
        self.anchorsCoords=[]
        self.nonAnchorMask=np.full(self.num*self.dim,True)
        self.anchorMask=np.full(self.num*self.dim,False)
        self.anchorEdge=[]
        self.nonAnchorEdge=[]
        self.X0=np.zeros(self.dim*self.num)
        self.nodes=[]
        self.edges=[]
        self.DELvect = np.zeros(self.G.number_of_edges())
        self.fitDist = np.zeros(self.G.number_of_edges())
        self.translate= False
        for n in self.G.nodes():
            self.nodes.append(n)
            if n in self.anchored:
                self.anchors.append(n)
                self.anchorsCoords.append(self.G.nodes[n]['Coords'])
        ind = 0
        self.incMat_s=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.incMat_t=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.dist0=np.zeros(self.G.number_of_nodes())
        self.sources=[]
        self.targets=[]
        oorigin = np.zeros(self.dim + 1)
        oorigin[-1] = 1
        if self.anchors :
            if len(self.anchors) < self.G.number_of_nodes():
                #we need to fit
                self.toFit=True
                anchorsCoords = np.array(self.anchorsCoords)
                if origin is None:
                    self.origin = self.anchors[0]
                else:
                    if origin in self.anchors:
                        self.origin = origin
                    else:
                        self.origin=self.anchors[0]
                if not (np.array_equal(origin, self.anchorsCoords[0])):
                    #we need to translate the choosen vertex to the origin
                    self.translate = True
                    self.PP = anchorsCoords[0]
                    anchorsCoords = self.translationToOrigin(anchorsCoords, np.array(anchorsCoords[0]))
#                    ind +=1
                    for anchor in self.anchors:
                        I = self.nodes.index(anchor)
                        self.anchorMask[I::self.num] = True
                        self.nonAnchorMask[I::self.num] = False
                        self.X0[I::self.num] = anchorsCoords[ind,0:-1]
                        ind +=1
            else:
                # all vertices are already anchored
                # we have nothing to do
                ind = 0
                for u, v, data in self.G.edges(data=True):
                    self.edges.append((u, v))
                    self.DELvect[ind] = data['distance']
                    I = self.nodes.index(u)
                    J = self.nodes.index(v)
                    self.sources.append(I)
                    self.targets.append(J)
                    self.incMat_s[I, ind] = 1
                    self.incMat_t[J, ind] = 1
                    if (u in self.anchors) and (v in self.anchors):
                        self.anchorEdge.append(ind)
                    else:
                        self.nonAnchorEdge.append(ind)
                    ind +=1
                self.toFit=False
        else:
            self.toFit=True
            if oorigin is None:
                self.origin = list(self.G.nodes)[0]
            else:
                if origin is None:
                    self.origin = list(self.G.nodes)[0]
                else:
                    self.origin = origin
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1

            self.fullGraph.nodes[self.origin]['Coords'] = oorigin
            self.G.nodes[self.origin]['Coords'] = oorigin
            self.anchorMask[0::self.num] = True
            self.nonAnchorMask[0::self.num] = False
            self.X0[0::self.num] = oorigin[0:-1]
            self.anchors.append(self.origin)
            self.translate = False
        if self.toFit:
            ind=0
            for u, v, data in self.G.edges(data=True):
                self.edges.append((u, v))
                self.DELvect[ind] = data['distance']
                I = self.nodes.index(u)
                J = self.nodes.index(v)
                self.sources.append(I)
                self.targets.append(J)
                self.incMat_s[I, ind] = 1
                self.incMat_t[J, ind] = 1
                if (u in self.anchors) and (v in self.anchors):
                    self.anchorEdge.append(ind)
                else:
                    self.nonAnchorEdge.append(ind)
                ind += 1


    def hdistmat(self, Xsources, Xtargets,K=None):
        #Xsources is a matrix with n rows and k+1 columns
        #Xtargets is a matrix with n rows and k+1 columns
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        if K is None:
            K=self.K
        if np.where(B>-1):
            B[np.where(B>-1)]=-1
        D=(K**2)*np.arccosh(-B)
        return D

    def Bmat(self,Xsources, Xtargets):
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        return B

    def translationToOrigin(self,X,PP):
    # Calculate the new positions of X when origin move to position P
        P=PP.copy()
        P[0:-1]=-P[0:-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.matmul(mat, X.transpose())
        return translated.transpose()

    def translationBack(self,X,PP):
        P=PP.copy()
        P[-1]=P[-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.dot(mat, X.transpose())
        return translated.transpose()

    def PD_cost(self,X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        Xsources=X[sources][self.nonAnchorEdge]
        Xtargets=X[targets][self.nonAnchorEdge]
        D=self.hdistmat(Xsources, Xtargets)
        dd=d[self.nonAnchorEdge]
        F = np.sum((D- dd)** 2)
        return F

    def PD_gradient(self, X1, sources, targets, d):
        # X is a vector of nk values
        # XX is a matrix of k points in k+1 dimension and an array of k values
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        K=self.K
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1]**2, axis=1)+1)
        Xsources=X[sources,:][self.nonAnchorEdge]
        Xtargets=X[targets,:][self.nonAnchorEdge]
        D=self.hdistmat(Xsources, Xtargets,K)
        dd=d[self.nonAnchorEdge]
        err=D-dd
        B=self.Bmat(Xsources, Xtargets)
        if np.where(B>-1)[0].size>0:
            print("problem")
        dE=np.zeros((num,self.dim))
        coupling_s = Xtargets[:,0:-1]-((Xtargets[:,-1]/Xsources[:,-1])*Xsources[:,0:-1].T).T
        coupling_t = Xsources[:,0:-1]-((Xsources[:,-1]/Xtargets[:,-1])*Xtargets[:,0:-1].T).T
        coef=-2*K**2*err/np.sqrt(np.maximum(B**2-1,1e-16))
        val_s=(coef*coupling_s.T).T
        val_t=(coef*coupling_t.T).T
        for k in range(self.dim):
            dE[:,k]=self.incMat_s[:,self.nonAnchorEdge]@val_s[:,k]+self.incMat_t[:,self.nonAnchorEdge]@val_t[:,k]
        dEE=dE.T.reshape(-1).ravel()[self.nonAnchorMask]
        return dEE

    def HYP_jacobian(self, X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0, :] = np.zeros((1, self.dim))
        Xsources=X[sources]
        Xtargets=X[targets]
        dX=Xtargets-Xsources
        F = (np.sum(dX ** 2, axis=1) - d ** 2)
        EE = np.zeros((num*self.dim, num*self.dim))
        for k in range(self.dim):
            FF = np.zeros((len(sources), num*self.dim))
            for l in range(self.dim):
                for m in range(len(sources)):
                    if l == k:
                        FF[m, sources[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                        FF[m, targets[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                    else:
                        FF[m, sources[m] + l * num]=2*dX[m,k]*dX[m,l]
                        FF[m, targets[m] + l * num]=2*dX[m,k]*dX[m,l]
            E = self.incMat.dot(FF)
            EE[k*num:(k+1)*num, :] = E
#        EE[0,:].fill(0)
#        EE[:,0].fill(0)
        return 4*EE[np.ix_(self.nonAnchorMask,self.nonAnchorMask)]



    def HYP_fit(self,X0=None):
#        np.random.seed(0)
        if self.toFit :
            numVar=np.sum(self.nonAnchorMask)//self.dim
            if X0 is None:
                numNode=self.nonAnchorMask.size//self.dim
                X0 = np.random.rand(numNode, self.dim)
            XX0=X0.T.reshape(-1)[self.nonAnchorMask]
            roots = minimize(self.HYP_cost, XX0.reshape((1, numVar*self.dim), order='F').ravel(),
                            args=(self.sources, self.targets, self.DELvect), method='L-BFGS-B', jac=self.HYP_gradient,#hess=self.MDS_jacobian,
                             options={'disp': True, 'maxiter': 1000, 'maxiter':5000})
           #hess = self.jacobian
    #        roots = minimize(self.cost, X0.reshape((1, self.G.number_of_nodes() * self.dim), order='F').ravel(),
    #                         args=(self.sources, self.targets, self.DELvect), method='Nelder-Mead', jac=self.equations
    #                         , options={'disp': True, 'maxiter': 1000, 'verbose': 3})
            XX = roots['x']
            X = self.X0.copy()
            X[self.nonAnchorMask] = XX
            X = X.reshape((self.G.number_of_nodes(), self.dim), order='F')
            self.extendtoHyperboloid(X)

        else:
            self.coords=np.array(self.anchorsCoords)
        Xsources = self.coords[self.sources, :][self.nonAnchorEdge]
        Xtargets = self.coords[self.targets, :][self.nonAnchorEdge]
        self.D=self.hdistmat(Xsources, Xtargets)
        self.delta=self.D-self.DELvect[self.nonAnchorEdge]
        self.sserror=np.sum(self.delta**2)
        if self.G.number_of_edges() > 0:
            self.rmse=np.sqrt(self.sserror/self.G.number_of_edges())
        else:
            self.rmse=0.0
        if self.rmse != self.rmse :
            pass

        self.rerror=np.sum((self.delta/self.DELvect[self.nonAnchorEdge])**2)
        for u,v,data in self.G.edges(data=True):
            self.G[u][v]['edist'] = \
            self.hdistmat(np.array(self.G.nodes[u]['Coords'])[:, np.newaxis].transpose(),
                  np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                  self.K)[0]
            self.fullGraph[u][v]['edist'] = self.G[u][v]['edist']
        return self.coords,self.K

    def embed_graph(self,G=None,tree=None):
        if G is None:
            G=self.fullGraph

    def extendtoHyperboloid(self,X0):
        self.coords=np.zeros((X0.shape[0],self.dim+1))
        self.coords[:,0:-1]=X0
        self.coords[:,-1]=np.sqrt(np.sum(X0**2, axis=1)+1)
        if self.translate:
            self.coords=self.translationBack(self.coords,self.PP)
        for i in range(X0.shape[0]):
            self.fullGraph.nodes[self.nodes[i]]['Coords']=self.coords[i]
            self.fullGraph.nodes[self.nodes[i]]['anchored']=1
            self.G.nodes[self.nodes[i]]['Coords'] = self.coords[i]
            self.G.nodes[self.nodes[i]]['anchored']=1

        for u,v,data in self.G.edges(data=True):
            X1=Hyperpoint(self.G.nodes[u]['Coords'],self.C)
            X2=Hyperpoint(self.G.nodes[v]['Coords'],self.C)
            self.fullGraph[u][v]['estDist']=X1.hdist(X2)
            data['estDist']=X1.hdist(X2)

#class for hyperboloid gradient method
class HyperHypNew:
    def __init__(self,nodeSet, fullGraph, C, dim,  verb, dist0=None, origin=None, stress='rmse'):
        self.verb = verb
        self.fullGraph = fullGraph
        self.G=nx.subgraph(fullGraph,nodeSet).copy()
        self.num=self.G.number_of_nodes()
        self.C=C
        self.K=1/np.sqrt(C)
        self.dim=dim
        self.anchors=[]
        self.anchorsCoords=[]
        self.nonAnchorMask=np.full(self.num*self.dim,True)
        self.anchorMask=np.full(self.num*self.dim,False)
        self.X0=np.zeros(self.dim*self.num)
        self.nodes=[]
        self.edges=[]
        self.D0=np.zeros(self.G.number_of_edges())
        self.DELvect = np.zeros(self.G.number_of_edges())
        self.fitDist = np.zeros(self.G.number_of_edges())
        self.translate= False
        i=0
        for u,v,data in self.G.edges(data=True):
            self.D0[i]=data['distance']
            i +=1
        for n in self.G.nodes():
            self.nodes.append(n)
            if 'Coords' in self.G.nodes[n]:
                self.anchors.append(n)
                self.anchorsCoords.append(self.G.nodes[n]['Coords'])
        ind = 0
        self.incMat_s=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.incMat_t=np.zeros((self.G.number_of_nodes(),self.G.number_of_edges()))
        self.dis0=np.zeros(self.G.number_of_nodes())
        self.sources=[]
        self.targets=[]
#        origin = np.zeros(self.dim + 1)
#        origin[-1] = 1
        if self.anchors :
            if len(self.anchors) < self.G.number_of_nodes():
                #we need to fit
                self.toFit=True
                anchorsCoords = np.array(self.anchorsCoords)
                if origin is None:
                    self.origin = self.anchors[0]
                else:
                    if origin in self.anchor:
                        self.origin = origin
                    else:
                        self.origin = self.anchors[0]
                if not (np.array_equal(origin, self.anchorsCoords[0])):
                    #we need to translate the choosen vertex to the origin
                    self.translate = True
                    self.PP = anchorsCoords[0]
                    anchorsCoords = self.translationToOrigin(anchorsCoords, np.array(anchorsCoords[0]))
                    ind =0
                    for anchor in self.anchors:
                        I = self.nodes.index(anchor)
                        self.anchorMask[I::self.num] = True
                        self.nonAnchorMask[I::self.num] = False
                        self.X0[I::self.num] = anchorsCoords[ind,0:-1]
                        ind+=1
            else:
                # all vertices are already anchored
                # we have nothing to do
                for u, v, data in self.G.edges(data=True):
                    self.edges.append((u, v))
                    self.DELvect[ind] = data['distance']
                    I = self.nodes.index(u)
                    J = self.nodes.index(v)
                    self.sources.append(I)
                    self.targets.append(J)
                    self.incMat_s[I, ind] = 1
                    self.incMat_t[J, ind] = 1
                    self.DELvect[ind] = data['distance']
                self.toFit=False
        else:
            self.toFit=True
            if origin is None:
                self.origin = list(self.G.nodes)[0]
            else:
                self.origin = origin
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1
            self.fullGraph.nodes[self.origin]['Coords'] = oorigin
            self.G.nodes[self.origin]['Coords'] = oorigin
            self.anchorMask[0::self.num] = True
            self.nonAnchorMask[0::self.num] = False
            self.X0[0::self.num] = oorigin[0:-1]
            self.anchors.append(self.origin)
            self.translate = False
        if self.toFit:
            for u, v, data in self.G.edges(data=True):
                self.edges.append((u, v))
                I = self.nodes.index(u)
                J = self.nodes.index(v)
                self.sources.append(I)
                self.targets.append(J)
                self.incMat_s[I, ind] = 1
                self.incMat_t[J, ind] = 1
                self.DELvect[ind] = data['distance']
                self.fitDist[ind] = 2 * np.cosh(data['distance'] / self.K**2) + (
                        np.cosh(dist0[I] / self.K**2) - np.cosh(dist0[J] / self.K**2)) ** 2 - 2
                ind += 1

        #        if stress == 'rmse':
#            self.fgrad = self.fgrad_rmse
#            self.fstress = self.fstress_rmse
#        elif stress == 'rrmse':
#            self.fgrad = self.fgrad_rrmse
#            self.fstress=self.fstress_rrmse
#        elif stress == 'sam':
#            self.fgrad = self.fgrad_sam
#            self.fstress=self.fstress_sam
#        self.translate = False

    def hdistmat(self, Xsources, Xtargets, K=None):
        #Xsources is a matrix with n rows and k+1 columns
        #Xtargets is a matrix with n rows and k+1 columns
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        B[B>-1]=-1
        if K==None:
            K=self.K
        D=(K**2)*np.arccosh(-B)
        return D

    def Bmat(self,Xsources, Xtargets):
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        return B

    def translationToOrigin(self,X,PP):
    # Calculate the new positions of X when origin move to position P
        P=PP.copy()
        P[0:-1]=-P[0:-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.matmul(mat, X.transpose())
        return translated.transpose()

    def translationBack(self,X,PP):
        P=PP.copy()
        P[-1]=P[-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.dot(mat, X.transpose())
        return translated.transpose()

    def HYP_cost(self,X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1[0:-1]
        K=X1[-1]
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1]**2, axis=1)+1)
        Xsources=X[sources]
        Xtargets=X[targets]
        D=self.hdistmat(Xsources, Xtargets,K)
        Utils.plot(D,D-d)
        F = np.sum((D**2- d**2)** 2)
        return F

    def HYP_gradient(self, X1, sources, targets, d):
        # X is a vector of nk values
        # XX is a matrix of k points in k+1 dimension and an array of k values
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1[0:-1]
        K=X1[-1]
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:,0:-1]=X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1]**2, axis=1)+1)
        Xsources=X[sources,:]
        Xtargets=X[targets,:]
        D=self.hdistmat(Xsources, Xtargets,K)
        err=D**2-d**2
        B=self.Bmat(Xsources, Xtargets)
        dE=np.zeros((num,self.dim))
        coupling_s = X[targets,0:-1]-((X[targets,-1]/X[sources,-1])*X[sources,0:-1].T).T
        coupling_t = X[sources,0:-1]-((X[sources,-1]/X[targets,-1])*X[targets,0:-1].T).T
        coef=-2*K**2*err*D/np.sqrt(np.maximum(B**2-1,1e-9))
        val_s=(coef*coupling_s.T).T
        val_t=(coef*coupling_t.T).T
        for k in range(self.dim):
            dE[:,k]=self.incMat_s@val_s[:,k]+self.incMat_t@val_t[:,k]
        dK=np.sum(4*D**2*err/K)
        dEE=np.hstack((X0.reshape(1,self.dim*num).ravel()[self.nonAnchorMask],dK))
#        dEE = dE.T.reshape(-1).ravel()[self.nonAnchorMask]
        print(np.sum(err**2), np.linalg.norm(dEE), np.linalg.norm(dEE[0:-1]))
        return dEE

    def HYP_jacobian(self, X1, sources, targets, d):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0, :] = np.zeros((1, self.dim))
        Xsources=X[sources]
        Xtargets=X[targets]
        dX=Xtargets-Xsources
        F = (np.sum(dX ** 2, axis=1) - d ** 2)
        EE = np.zeros((num*self.dim, num*self.dim))
        for k in range(self.dim):
            FF = np.zeros((len(sources), num*self.dim))
            for l in range(self.dim):
                for m in range(len(sources)):
                    if l == k:
                        FF[m, sources[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                        FF[m, targets[m] + l * num] = F[m] + 2 * dX[m, l] ** 2
                    else:
                        FF[m, sources[m] + l * num]=2*dX[m,k]*dX[m,l]
                        FF[m, targets[m] + l * num]=2*dX[m,k]*dX[m,l]
            E = self.incMat.dot(FF)
            EE[k*num:(k+1)*num, :] = E
#        EE[0,:].fill(0)
#        EE[:,0].fill(0)
        return 4*EE[np.ix_(self.nonAnchorMask,self.nonAnchorMask)]



    def HYP_fit(self,X0=None):
#        np.random.seed(0)
        if self.toFit :
            numVar=np.sum(self.nonAnchorMask)//self.dim
            if X0 is None:
                X0 = np.random.rand(numVar, self.dim)
            XX0=np.zeros(numVar * self.dim+1)
            XX0[0:-1]=X0.reshape((1, numVar * self.dim), order='F').ravel()
            XX0[-1]=1
            roots = minimize(self.HYP_cost, XX0,
                            args=(self.sources, self.targets, self.DELvect), method='Newton-CG' , jac=self.HYP_gradient,#hess=self.MDS_jacobian,
                             options={'disp': True, 'maxiter': 7000})
           #hess = self.jacobian
    #        roots = minimize(self.cost, X0.reshape((1, self.G.number_of_nodes() * self.dim), order='F').ravel(),
    #                         args=(self.sources, self.targets, self.DELvect), method='Nelder-Mead', jac=self.equations
    #                         , options={'disp': True, 'maxiter': 1000, 'verbose': 3})
            XX = roots['x']
            X = self.X0.copy()
            X[self.nonAnchorMask] = XX[0:-1]
            self.K=X[-1]
            X = X.reshape((self.G.number_of_nodes(), self.dim), order='F')
            self.extendtoHyperboloid(X)

        else:
            self.coords=np.array(self.anchorsCoords)
        Xsources = self.coords[self.sources, :]
        Xtargets = self.coords[self.targets, :]
        self.D=self.hdistmat(Xsources, Xtargets,self.K)
        self.delta=self.D-self.DELvect
        self.sserror=np.sum(self.delta**2)
        self.rmse=np.sqrt(self.sserror/self.G.number_of_edges())
        self.rerror=np.sum((self.delta/self.DELvect)**2)
        return [self.coords,self.K]

    def embed_graph(self,G=None,tree=None):
        if G is None:
            G=self.fullGraph

    def extendtoHyperboloid(self,X0):
        self.coords=np.zeros((X0.shape[0],self.dim+1))
        self.coords[:,0:-1]=X0
        self.coords[:,-1]=np.sqrt(np.sum(X0**2, axis=1)+1)
        if self.translate:
            self.coords=self.translationBack(self.coords,self.PP)
        for i in range(X0.shape[0]):
            self.fullGraph.nodes[self.nodes[i]]['Coords']=self.coords[i]
            self.fullGraph.nodes[self.nodes[i]]['anchored']=1

        for u,v,data in self.G.edges(data=True):
            X1=Hyperpoint(self.G.nodes[u]['Coords'],self.C)
            X2=Hyperpoint(self.G.nodes[v]['Coords'],self.C)
            self.fullGraph[u][v]['estDist']=X1.hdist(X2)
            self.G[u][v]['estDist']=X1.hdist(X2)

#class for Euclidean Gradient method
class HyperMDS:
    def __init__(self, nodeSet, MfullGraph, C, dim, rootNode=None, verb=1, stress='rmse'):
        self.verb = verb
        self.MetricGraph = MfullGraph
        self.fullGraph = self.MetricGraph.G
        #copy the subgraph
        self.G = nx.subgraph(self.fullGraph, nodeSet).copy()
        self.num = self.G.number_of_nodes()
        self.C = C
        self.K = 1 / np.sqrt(C)
        self.dim = dim
        self.anchors = []
        self.anchorsCoords = {}
        self.nonAnchorMask = np.full(self.num * self.dim, True)
        self.anchorMask = np.full(self.num * self.dim, False)
        self.X0 = np.random.rand( self.num*self.dim)
        self.D0 = np.full(self.G.number_of_edges(), np.inf)
        i=0
        for u,v,data in self.G.edges(data=True):
            self.D0[i]=data['distance']
            i +=1
        self.nodes = []
        self.edges = []
        self.translate = False
        for n in self.G.nodes():
            self.nodes.append(n)
            if 'anchored' in self.G.nodes[n] and self.G.nodes[n]['anchored']==1:
                self.anchors.append(n)
                self.anchorsCoords[n]=self.G.nodes[n]['Coords']
        ind = 0
        self.sources = []
        self.targets = []
        self.anchoredEdges = []
        self.unanchoredEdges = []
        self.toFit=True
        if self.anchors:
            #we know that there are node that are anchored.
            if len(self.anchors) < self.G.number_of_nodes():
                # we need to fit
                self.toFit = True
                if rootNode is None:
                    maxdeg=0
                    for n in self.anchors:
                        if self.G.degree[n] > maxdeg:
                            maxdeg = self.G.degree[n]
                            self.origin = n
                else:
                    if rootNode in self.anchors:
                        self.origin = rootNode
                    else:
                        maxdeg = 0
                        for n in self.anchors:
                            if self.G.degree[n] > maxdeg:
                                maxdeg = self.G.degree[n]
                                self.origin = n
            else:
                # all vertices are already anchored
                # we have nothing to do
                self.toFit = False
        else:
            if rootNode:
                self.origin = rootNode
                self.anchors.append(self.origin)
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1
                self.anchorsCoords[self.origin]=oorigin
            else:
                maxdeg = 0
                for n in self.G.nodes():
                    if self.G.degree[n] > maxdeg:
                        maxdeg = self.G.degree[n]
                        self.origin = n
                self.anchors.append(self.origin)
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1
                self.anchorsCoords[self.origin] = oorigin
            if (self.anchorsCoords[self.origin] == oorigin).all():
                self.translate = False
            else:
                # we need to translate the choosen vertex to the origin
                self.translate = True
                PP = self.anchorsCoords[self.origin]
                X = np.array(len(self.anchorsCoords))
                ind = 0
                for key, x in self.anchorsCoords.items():
                    X[ind] = x
                newanchorsCoords = self.translationToOrigin(self.anchors, PP)
                ind=0
                for key, x in self.anchorsCoords.items():
                    self.anchorsCoords[key] = newanchorsCoords[ind]
                    self.fullGraph.nodes[key]['Coords']=newanchorsCoords[ind]
                    self.G.nodes[key]['Coords']=newanchorsCoords[ind]
        #set the anchorMask, nonAnchorMasy
        for anchor in self.anchors:
            I=self.nodes.index(anchor)
            self.anchorMask[I::self.num] = True
            self.nonAnchorMask[I::self.num] = False
            self.X0[I::self.num] = self.anchorsCoords[anchor][0:-1]
        if self.toFit:
            self.dis0 = np.zeros(self.G.number_of_nodes())
            dist0=self.MetricGraph.getdist0(self.origin)
            # Add spurious link to the graph
            for n in self.G.nodes():
                if not self.G.has_edge(n,self.origin):
                    if n != self.origin:
                        self.G.add_edge(n,self.origin,spurious=True,distance=dist0[n])
            self.incMat = np.zeros((self.G.number_of_nodes(), self.G.number_of_edges()))
            self.incMat = np.zeros((self.G.number_of_nodes(), self.G.number_of_edges()))
            self.DELvect = np.full(self.G.number_of_edges(), np.inf)
            self.fitDist = np.full(self.G.number_of_edges(), np.inf)
            ind=0
            for u, v, data in self.G.edges(data=True):
                self.edges.append((u, v))
                I = self.nodes.index(u)
                J = self.nodes.index(v)
                self.sources.append(I)
                self.targets.append(J)
                self.incMat[I, ind] = -1
                self.incMat[J, ind] = 1
                self.DELvect[ind] = data['distance']
                self.fitDist[ind] = 2*np.cosh(self.DELvect[ind]/(self.K**2))-(np.cosh(dist0[u]/(self.K**2))-np.cosh(dist0[v]/(self.K**2)))**2-2
                if (u in self.anchors) and (v in self.anchors):
                    self.anchoredEdges.append(ind)
                else:
                    self.unanchoredEdges.append(ind)
                ind += 1

        #        if stress == 'rmse':

    #            self.fgrad = self.fgrad_rmse
    #            self.fstress = self.fstress_rmse
    #        elif stress == 'rrmse':
    #            self.fgrad = self.fgrad_rrmse
    #            self.fstress=self.fstress_rrmse
    #        elif stress == 'sam':
    #            self.fgrad = self.fgrad_sam
    #            self.fstress=self.fstress_sam
    #        self.translate = False

    def hdistmat(self, Xsources, Xtargets, K=None):
        #Xsources is a matrix with n rows and k+1 columns
        #Xtargets is a matrix with n rows and k+1 columns
        B=np.sum(Xsources[:,0:-1]*Xtargets[:,0:-1],axis=1)-Xsources[:,-1]*Xtargets[:,-1]
        B[B>-1]=-1
        if K is None:
            K=self.K
        D=K**2*np.arccosh(-B)
        return D

    def translationToOrigin(self,X,PP):
    # Calculate the new positions of X when origin move to position P
        P=PP.copy()
        P[0:-1]=-P[0:-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.matmul(mat, X.transpose())
        return translated.transpose()

    def translationBack(self,X,PP):
        P=PP.copy()
        P[-1]=P[-1]
        mat=np.zeros((P.shape[0],P.shape[0]))
        mat[0:-1,0:-1]=1/(P[-1]+1)*np.outer(P[0:-1],P[0:-1])+np.eye(P.shape[0]-1)
        mat[-1,:]=P
        mat[:,-1]=P
        translated=np.dot(mat, X.transpose())
        return translated.transpose()

    def MDS_cost_SRSE(self,X1):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0,:]=np.zeros((1,self.dim))
        Xsources=X[self.sources][self.unanchoredEdges]
        Xtargets=X[self.targets][self.unanchoredEdges]
        dX=Xtargets-Xsources
        FF=np.sum(dX ** 2,axis=1)/self.fitDist[self.unanchoredEdges]-1
        F = np.sum(FF**2)
        return F

    def MDS_cost_SSE(self,X1):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0,:]=np.zeros((1,self.dim))
        Xsources=X[self.sources][self.unanchoredEdges]
        Xtargets=X[self.targets][self.unanchoredEdges]
        dX=Xtargets-Xsources
        D=np.sum(dX ** 2, axis=1)
        FF=(D - self.fitDist[self.unanchoredEdges])
        F = np.sum(FF**2)
        return F

    def MDS_gradient_SRSE(self, X1):
        # X is a vector of nk values
        # XX is a matrix of k points in k+1 dimension and an array of k values
        X0 = self.X0.copy()
        X0[self.nonAnchorMask] = X1
        num = int(X0.shape[0] / self.dim)
        X = X0.reshape(num, self.dim, order='F')
        Xsources=X[self.sources]
        Xtargets=X[self.targets]
        dX = Xtargets - Xsources
        E = np.empty((self.dim, num))
        D = np.sum(dX ** 2, axis=1)
        F = (D - self.fitDist)/self.fitDist**2-1
        for k in range(self.dim):
            #            E[k,:]=8*self.incMat@(F*dX[:,k]*D)
            E[k, :] = 4 * self.incMat @ (F * dX[:, k])
        EE = E.reshape(1, X0.shape[0]).ravel()
        dEE = E.T.reshape(-1).ravel()[self.nonAnchorMask]
        return EE[self.nonAnchorMask]

    def MDS_gradient_SSE(self, X1):
        # X is a vector of nk values
        # XX is a matrix of k points in k+1 dimension and an array of k values
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        Xsources=X[self.sources,:]
        Xtargets=X[self.targets,:]
        dX=Xtargets-Xsources
        E=np.empty((self.dim,num))
        D=np.sum(dX ** 2, axis=1)
        F = (D- self.fitDist)
        for k in range(self.dim):
#            E[k,:]=8*self.incMat@(F*dX[:,k]*D)
            E[k, :] = 4 * self.incMat @ (F * dX[:, k])
        EE=E.reshape(1, X0.shape[0]).ravel()
#        EE = E.T.reshape(-1).ravel()[self.nonAnchorMask]
        return EE[self.nonAnchorMask]

    def MDS_jacobian(self, X1):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=X0.reshape(num,self.dim, order='F')
        X[0, :] = np.zeros((1, self.dim))
        Xsources=X[self.sources]
        Xtargets=X[self.targets]
        dX=Xtargets-Xsources
        D=np.sqrt(np.sum(dX ** 2, axis=1))
        F = (D - self.fitDist)
        EE = np.zeros((num*self.dim, num*self.dim))
#        dXX=dX.reshape(num**2*self.dim**2,order='F')
#        coupling=np.outer(dXX,dXX)
#        for k in range(self.dim):
#            EE[]
#        E[sources,targets]
        for m in range(len(self.sources)):
            for k in range(self.dim):
                for l in range(self.dim):
                    if l == k:
#                        EE[sources[m] + l * num][targets[m]+k*num] = -8*(F[m]*dX[m, l]**2/D[m]  + 2 * dX[m, l] ** 2*D[m]+F[m]*D[m])
#                        EE[targets[m]+k*num][sources[m] + l * num] = EE[sources[m] + l * num][targets[m]+k*num]
                        EE[self.sources[m] + l * num][self.targets[m]+k*num] = -4*(F[m]  + 2 * dX[m, l]**2)
                        EE[self.targets[m]+k*num][self.sources[m] + l * num] = EE[self.sources[m] + l * num][self.targets[m]+k*num]
                    else:
#                        EE[sources[m] + l * num][targets[m]+k*num] = -8*(F[m]*dX[m, l]*dX[m,k]/D[m]+2*D[m]*dX[m,l]*dX[m,k])
#                        EE[targets[m]+k*num][sources[m] + l * num] = EE[sources[m] + l * num][targets[m]+k*num]
                        EE[self.sources[m] + l * num][self.targets[m]+k*num] = -8*dX[m, l]*dX[m,k]
                        EE[self.targets[m]+k*num][self.sources[m] + l * num] = EE[self.sources[m] + l * num][self.targets[m]+k*num]

#        EE[0,:].fill(0)
#        EE[:,0].fill(0)
        return EE[np.ix_(self.nonAnchorMask,self.nonAnchorMask)]

    def MDS_constraint(self, X1):
        X0=self.X0.copy()
        X0[self.nonAnchorMask]=X1
        num=int(X0.shape[0] / self.dim)
        X=np.zeros((num,self.dim+1))
        X[:, 0:-1] =X0.reshape(num,self.dim, order='F')
        X[:,-1]=np.sqrt(np.sum(X[:,0:-1] ** 2, axis=1) + 1)
        Xsources=X[self.sources][self.unanchoredEdges]
        Xtargets=X[self.targets][self.unanchoredEdges]
        D=self.hdistmat(Xsources, Xtargets)
        return(np.mean(self.DELvect-D))

    def MDS_fit(self,X0=None):
    #        np.random.seed(0)
        if self.toFit:
            numVar = np.sum(self.nonAnchorMask) // self.dim
            if X0 is None:
                numNode = self.nonAnchorMask.size // self.dim
                X0 = np.random.rand(numNode, self.dim)
            X1 = X0.T.reshape(-1)[self.nonAnchorMask]
            roots = minimize(self.MDS_cost_SSE, X1.reshape((1, numVar*self.dim), order='F').ravel(),
                             method="L-BFGS-B", jac=self.MDS_gradient_SSE, #hess=self.MDS_jacobian,
                             #constraints={'type':'eq','fun':self.MDS_constraint},
                             options={'disp': True, 'maxiter': 50000})
           #hess = self.jacobian
    #        roots = minimize(self.cost, X0.reshape((1, self.G.number_of_nodes() * self.dim), order='F').ravel(),
    #                         args=(self.sources, self.targets, self.DELvect), method='Nelder-Mead', jac=self.equations
    #                         , options={'disp': True, 'maxiter': 1000, 'verbose': 3})
            XX = roots['x']
            X = self.X0.copy()
            X[self.nonAnchorMask] = XX
            X = X.reshape((self.G.number_of_nodes(), self.dim), order='F')
            self.extendtoHyperboloid(X)
        else:
            self.coords=np.array(self.anchorsCoords)
        if self.unanchoredEdges :
            Xsources = self.coords[self.sources, :]
            Xtargets = self.coords[self.targets, :]
            nonSpuriousEdges=[]
            for m in self.unanchoredEdges:
                if "spurious" not in self.G[self.nodes[self.sources[m]]][self.nodes[self.targets[m]]]:
                    nonSpuriousEdges.append(m)
            self.D=self.hdistmat(Xsources[nonSpuriousEdges], Xtargets[nonSpuriousEdges], self.K)
            self.delta=self.D-self.DELvect[nonSpuriousEdges]
            self.sserror=np.sum(self.delta**2)
            self.rmse=np.sqrt(self.sserror/len(nonSpuriousEdges))
        else:
            self.D=[]
            self.delta=[]
            self.sserror=0.0
            self.rmse=0.0
        return self.coords
#        self.rerror=np.sum((self.delta/self.DELvect)**2)

    def embed_graph(self,G=None,tree=None):
        if G is None:
            G=self.fullGraph

    def extendtoHyperboloid(self,X0):
        self.coords=np.zeros((X0.shape[0],self.dim+1))
        self.coords[:,0:-1]=X0
        self.coords[:,-1]=np.sqrt(np.sum(X0**2, axis=1)+1)
        if self.translate:
            self.coords=self.translationBack(self.coords,self.PP)
        for i in range(X0.shape[0]):
            self.fullGraph.nodes[self.nodes[i]]['Coords']=self.coords[i]
            self.fullGraph.nodes[self.nodes[i]]['anchored']=1
            self.G.nodes[self.nodes[i]]['Coords']=self.coords[i]
            self.G.nodes[self.nodes[i]]['anchored']=1
        for u,v,data in self.G.edges(data=True):
            if self.fullGraph.has_edge(u,v):
                X1=Hyperpoint(self.fullGraph.nodes[u]['Coords'],self.C)
                X2=Hyperpoint(self.fullGraph.nodes[v]['Coords'],self.C)
                self.fullGraph[u][v]['estDist']=X1.hdist(X2)





class HyperHyper:
    def __init__(self):
        stress='rmse'
        if stress == 'rmse':
            self.fgrad = self.fgrad_rmse
            self.fstress = self.fstress_rmse
        elif stress == 'rrmse':
            self.fgrad = self.fgrad_rrmse
            self.fstress=self.fstress_rrmse
        elif stress == 'sam':
            self.fgrad = self.fgrad_sam
            self.fstress=self.fstress_sam

    def equations(self, X,XX0,a):#XX is a matrix of k points in k+1 dimension and a an array of k values
        XX=np.append(X,-np.sqrt(np.sum(X*X)+1))
        aa=np.matmul(XX0, XX.transpose())
        F= aa-a
        return F

    def jacobian(self, X,XX0,a):
        XX = X/np.sqrt(np.sum(X * X) + 1)
        PP=np.outer(XX0[:,-1], XX)
        prime=XX0[:,0:-1]-PP
#        prime[0,0]=X0[0]-X0[2]*X2[0]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[0,1]=X0[1]-X0[2]*X2[1]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[1,0]=X1[0]-X1[2]*X2[0]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[1,1]=X1[1]-X1[2]*X2[1]/np.sqrt(X2[0]**2+X2[1]**2+1)
        return prime

    def embed(self, X0, a, alpha):
        X0=mp.matrix(X0).transpose()
        r=mp.sqrt(a**2-1)
        mp.matrix(3)
        X=r*mp.cos(alpha)
        Y=r*mp.sin(alpha)
        Z=a
        XX=mp.matrix([[X,Y,Z]])
        if (mp.almosteq(X0[2],1)):
            if mp.fabs(XX[0] ** 2 + XX[1] ** 2 - XX[2] ** 2 + 1) > 1e-1:
                pass
            return(np.array(XX.tolist(),dtype=np.float64).ravel())
        else:
            u=X0
            u[2] -=1
            u=u/mp.sqrt(u[0]**2+u[1]**2-u[2]**2)
            X1=XX-2*(X*u[0]+Y*u[1]-Z*u[2])*u
            X1 = X1.apply(mp.re)
            return (np.array(X1.tolist(), dtype=np.float64).ravel())


    def hypdist(self,Xs,Xt,C):
        K=1/np.sqrt(C)
        a=np.sum(Xs[:,0:-1]*Xt[:,0:-1], axis=1)-Xs[:,-1]*Xt[:,-1]
        a[np.where(a > -1)]=-1-1e-6
        D=K*np.arccosh(-a)
        return D

    def NN(self, coords, i,C):
        num=coords.shape[0]
        X0 =np.repeat(np.array(coords[i,:][np.newaxis]),repeats=num,axis=0)
        D=self.hypdist(X0,coords,C)
        return np.argsort(D)

    def evaluate(self,G,coords,C):
        for u in G.nodes():
            neigh=set(G.neighbors(u))
            rank=self.NN(coords, u, C).tolist()
            rank.remove(u)
            neighrank=[]
            for v in neigh:
                neighrank.append((v, rank.index(v)))
            neighrank.sort(key=lambda tup:tup[1])
            pass



    def sarkar_tree(self, treeGraph, root, C, **kwargs):
        tree = treeGraph
        K=1/np.sqrt(C)
        eps = kwargs.get("epsilon", 0.1)
        weighted = kwargs.get("weighted", True)
        tau = kwargs.get("tau")
        #    if tau is None:
        #        tau = (1 + eps) / eps * mpm.log(2 * max_deg / mpm.pi)
        prc = kwargs.get("precision")
        #    if prc is None:
        #        prc = tp._embedding_precision(tree, root, eps)
        #   mpm.mp.dps = prc
        angle =0.0
        deg=tree.degree[root]
        place=[]
        for v in tree[root]:
            tree[root][v]["angle"] = (np.pi) /(deg)
            tree[root][v]["dir"] = angle + 2*tree[root][v]["angle"]
            angle += 2*np.pi /deg
            place.append((root,v,tree[root][v]["dir"] ))
        while place:
            father,u,inangle= place.pop()
            deg=tree.degree[u]
            angle=inangle-np.pi
            for v in tree[u]:
                if v != father:
#                    tree[u][v]["angle"] = 2 * np.arctan(np.exp(-tree[u][v]["distance"] / k))
                    tree[u][v]["angle"]=np.pi/deg
                    tree[u][v]["dir"] = angle + 2*tree[u][v]["angle"]
                    place.append((u, v,tree[u][v]["dir"]))
                angle += 2 * np.pi/(deg)
                if angle > 2*np.pi:
                    angle -= 2*np.pi

        tree.nodes[root]["coord"] = [0.0, 0.0, 1.0]
        etamin = 1.0
        coords = np.zeros((tree.number_of_edges() + 1, 3))
        place = []
        #add the children of the root
        for v in tree[root]:
            place.append((root, v))
        minerr=10
        minarg=(0,0)
        while place:
            u, v = place.pop()  # u is the parent of v
            a = np.cosh(etamin * tree[u][v]["distance"]/K**2)
            alpha = tree[u][v]["dir"]
            X0 = tree.nodes[u]["coord"]
            X = self.embed(X0, a, alpha)
            tree.nodes[v]["coord"] = X
            treeGraph.nodes[v]["coord"]=X
#            d = K*np.arccosh(-(X0[0] * X[0] + X0[1] * X[1] - X0[2] * X[2]))
#            err=d - etamin * tree[u][v]["distance"]
#            if (abs(err)<minerr):
#                minerr=abs(err)
#                minarg=(u,v)
#                dmin=etamin * tree[u][v]["distance"]
#                amin=X0[0] * X[0] + X0[1] * X[1] - X0[2] * X[2]
#            print("d(", u, ",", v, ")=", d - etamin * tree[u][v]["distance"], tree[u][v]["distance"])
            for w in tree[v]:
                if (w != u):
                    place.append((v, w))
        cnt = 0
        for v in tree.nodes:
            coords[cnt, 0] = tree.nodes[v]["coord"][0]
            coords[cnt, 1] = tree.nodes[v]["coord"][1]
            coords[cnt, 2] = tree.nodes[v]["coord"][2]
            cnt += 1
        return treeGraph, coords

    def trianglecheck(self, a, b, c):
        if (a+b>=c) and (a+c>=b) and (b+c>=a):
            return True
        else:
            return False

    def hyperbolicCosLaw(self, a,b,c, K):
        if self.trianglecheck(a,b,c):
            if (c+b==a):
                alpha = np.pi
            else:
                if (np.cosh(b/K)*np.cosh(c/K)-np.cosh(a/K))/(np.sinh(b/K)*np.sinh(c/K))>1.0:
                    alpha = 0.0
                else:
                    alpha=np.arccos((np.cosh(b/K)*np.cosh(c/K)-np.cosh(a/K))/(np.sinh(b/K)*np.sinh(c/K)))
            return alpha
        else:
            if (c+b<a):
                return np.pi
            else :
                return 0.0


    def greedyEmbedding(self,tree, root,G,C):
        warnings.filterwarnings("error")
        nx.set_node_attributes(tree, False, "anchored")
        nx.set_node_attributes(tree, np.inf, "dir")
        count=0
        dcount=0
        K=1/np.sqrt(C)
        triangles=nx.triangles(G)
        processQueue=[]
        tree.nodes[root]["Coords"]=np.array([0,0,1])
        G.nodes[root]["Coords"]=np.array([0,0,1])
        tree.nodes[root]["anchored"]=True
        tree.nodes[root]["angle"]=0.0
        for u in tree[root]:
            processQueue.append((u,root,G.degree(u)))
        while processQueue:
            processQueue.sort(key=lambda tup: tup[2], reverse=True)
            neigh=set()
            fullneigh=[]
            w,father,d= processQueue.pop(0)# w is the point to position
            W = set(list(G.neighbors(w)))
            U=set()
            for u in W:
                if tree.nodes[u]["anchored"]:
                    U =set(list(G.neighbors(u)))
                    UW=U&W
                    for v in UW: # (father,w,v) is a triangle
                        if v not in neigh and tree.nodes[v]["anchored"] and G.has_edge(u,v):
                            neigh.add(v)
                            fullneigh.append((v,u,G[w][v]["distance"]))
#                            fullneigh.append((v,u,G.degree[v]))
            fullneigh.sort(key = lambda tup:-tup[2]) #full neigh contains all triangles based on w that have another node angchored
            if len(fullneigh)>1: # we can position w by triangylation
                u = fullneigh[0][1]
                v = fullneigh[0][0]
                count +=1
                X0 = tree.nodes[u]["Coords"]
                X1 = tree.nodes[v]["Coords"]
                a0 = -np.cosh(G[u][w]["distance"] / K**2)
                a1 = -np.cosh(G[v][w]["distance"] / K**2)
                XX0 = np.stack((X0, X1), axis=0)
                aa = np.array([a0, a1])
                sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,
                                    method='lm')
                roots = sol.x
                X2 = np.array([roots[0], roots[1], np.sqrt(roots[0] ** 2 + roots[1] ** 2 + 1)])
                tree.nodes[w]["Coords"] = X2
                G.nodes[w]["Coords"] = X2
#                aa0=X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]
#                aa1=X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]
#                d = K * np.arccosh(-(X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]))
#                d1= K * np.arccosh(-(X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]))
#                print(aa0-a0, aa1-a1)
#                print(G[u][w]["distance"] - d, G[v][w]["distance"]-d1)

                alpha = self.hyperbolicCosLaw(G[v][w]["distance"], G[u][v]["distance"], G[u][w]["distance"], K)
                tree.nodes[u]["angle"] += alpha
                if tree.has_edge(u,w):
                    tree[u][w]['dir'] = tree.nodes[u]["angle"]
                tree.nodes[w]["angle"] = np.pi - tree.nodes[u]["angle"]
            else:
                a = np.cosh(tree[father][w]["distance"] / K**2)
                alpha = np.pi / G.degree[father]
                X0 = tree.nodes[father]["Coords"]
                X = self.embed(X0, a, alpha)
                tree.nodes[w]["Coords"] = X
                G.nodes[w]["Coords"] = X
                #                        d=K*np.arccosh(-(X[0]*X0[0]+X[1]*X0[1]-X[2]*X0[2]))
                #                        print(tree[father][w]["distance"]-d,)
                tree.nodes[father]["angle"] += alpha
                tree[father][w]['dir'] = tree.nodes[father]["angle"]
                tree.nodes[w]["angle"] = np.pi - G.nodes[father]["angle"]
                dcount +=1
            tree.nodes[w]["anchored"] = True
            for t in tree[w]:
                if not tree.nodes[t]["anchored"]:
#                    processQueue.append((t, tree.nodes[t]["father"], tree[t][w]['distance']))
                    processQueue.append((t, tree.nodes[t]["father"], triangles[t]))

        cnt = 0
        coords = np.zeros((tree.number_of_nodes(), 3))
        for v in tree.nodes:
            coords[cnt, 0] = tree.nodes[v]["Coords"][0]
            coords[cnt, 1] = tree.nodes[v]["Coords"][1]
            coords[cnt, 2] = tree.nodes[v]["Coords"][2]
            cnt += 1
        print("count=", count, "dcount=", dcount)
        return tree, coords

    def greedyEmbedding2(self, G, C):
        warnings.filterwarnings("error")
        nx.set_node_attributes(G, np.inf, "dir")
        count = 0
        dcount = 0
        K = 1 / np.sqrt(C)
        triangles=nx.triangles(G)
        root=max(triangles, key=triangles.get)
        anchored=set()
        simplices=[]
        processQueue=[]
#        degree_sequence = sorted(G.degree(), key=lambda tup:tup[1], reverse=True)
#        root=degree_sequence[0][0]
        w=root
        G.nodes[root]["Coords"] = [0, 0, 1]
        G.nodes[root]["angle"] = 0.0
        anchored.add(root)
        simplex=Simplex([root])
        simplices.append(simplex)
        W = set(list(G.neighbors(w)))
        for u in W:
            processQueue.append((u,w,G.degree[u]))
        while processQueue:
            processQueue.sort(key=lambda tup: tup[2])
            simplices=sorted(simplices, key=lambda tup: tup.getDim())
            triangulation = False
            w, father, d = processQueue.pop(0)# w is the point to position
            if (w not in anchored):
                if (w==660):
                    pass
                potSimplices=[]
                for simplex in simplices:
                    anchors=simplex.check(w,G)
                    if (len(anchors)>0):
                        potSimplices.append((anchors,simplex))
                potAnchors=sorted(potSimplices, key=lambda tup: 1000*len(tup[0])-(tup[1].getDim()))
                chosenAnchors=potAnchors[0]
                if len(chosenAnchors[0])==1:
                    father=chosenAnchors[0][0]
                    a = np.cosh(G[father][w]["distance"] / K**2)
                    alpha = np.pi / G.degree[father]
                    X0 = G.nodes[father]["Coords"]
                    X = self.embed(X0, a, alpha)
                    G.nodes[w]["Coords"] = X
                    G.nodes[w]["Coords"] = X
                    #                        d=K*np.arccosh(-(X[0]*X0[0]+X[1]*X0[1]-X[2]*X0[2]))
                    #                        print(tree[father][w]["distance"]-d,)
                    G.nodes[father]["angle"] += alpha
                    G[father][w]['dir'] = G.nodes[father]["angle"]
                    G.nodes[w]["angle"] = np.pi - G.nodes[father]["angle"]
                    dcount += 1
                    if chosenAnchors[1].getDim()==len(chosenAnchors[0]):
                        chosenAnchors[1].add(w)
                    else:
                        simplex=Simplex(chosenAnchors[0])
                        simplex.add(w)
                        simplices.append(simplex)
                    anchored.add(w)
                else:# we can position w by triangylation
                    if chosenAnchors[1].getDim()==len(chosenAnchors[0]):
                        chosenAnchors[1].add(w)
                    else:
                        simplex=Simplex(chosenAnchors[0])
                        simplex.add(w)
                        simplices.append(simplex)
                    anchored.add(w)
                for t in G[w]:
                    if t not in anchored:
                        #                    processQueue.append((t, tree.nodes[t]["father"], tree[t][w]['distance']))
                        processQueue.append((t, w, G.degree[t]))

        for simplex in simplices:
            n=simplex.getDim()
            count += n*(n-1)/2

        cnt = 0
        coords = np.zeros((G.number_of_nodes(), 3))
        for v in G.nodes:
            coords[cnt, 0] = G.nodes[v]["Coords"][0]
            coords[cnt, 1] = G.nodes[v]["Coords"][1]
            coords[cnt, 2] = G.nodes[v]["Coords"][2]
            cnt += 1
        print("count=", count, "dcount=", dcount)
        return G, coords

    def greedyEmbedding3(self, G, C):
        warnings.filterwarnings("error")
        nx.set_node_attributes(G, np.inf, "dir")
        count = 0
        dcount = 0
        K = 1 / np.sqrt(C)
        triangles = nx.triangles(G)
        root = max(triangles, key=triangles.get)
        anchored = set()
        processQueue = []
        #        degree_sequence = sorted(G.degree(), key=lambda tup:tup[1], reverse=True)
        #        root=degree_sequence[0][0]
        w = root
        W = set(list(G.neighbors(w)))
        triangleDict = dict()
        triangleDict[w]=[]
        for u in W:
            U = set(list(G.neighbors(u)))
            UW = U & W
            for v in UW:
                triangleDict[w].append((u,v,w))
        G.nodes[root]["Coords"] = np.array([0, 0, 1])
        G.nodes[root]["angle"] = 0.0
        anchored.add(root)
        for u in G[root]:
            if len(triangleDict[u]) > 0:
                processQueue.append((u, w, counter[u] / 2))
        #            processQueue.append((u, root, triangles[u]))
        #            processQueue.append((u,root,G.degree[u]))
        while processQueue:
            processQueue.sort(key=lambda tup: tup[2], reverse=True)
            triangulation = False
            fullneigh = []
            w, father, d = processQueue.pop(0)  # w is the point to position
            if (w not in anchored):
                W = set(list(G.neighbors(w)))
                triangleDict= dict()
                counter = dict()
                for u in W:
                    counter[u] = 0
                    triangleDict[u] = []
                    # count triangles that are based on (u,w)
                    U = set(list(G.neighbors(u)))
                    UW = U & W
                    for v in UW:
                        if G.has_edge(u, v):  # (u,v,w) is a triangle
                            counter[u] += 1
                            if ((u in anchored) and (v in anchored)):
                                triangulation = True
                                triangleDict[u].append((u, v, w))
                counterList = sorted(counter, key=counter.get, reverse=True)
                if triangulation:  # we can position w by triangylation
                    for u in counterList:
                        if triangleDict[u]:
                            u = triangleDict[u][0][0]
                            v = triangleDict[u][0][1]
                            count += 1
                            X0 = G.nodes[u]["Coords"]
                            X1 = G.nodes[v]["Coords"]
                            a0 = -np.cosh(G[u][w]["distance"] / K**2)
                            a1 = -np.cosh(G[v][w]["distance"] / K**2)
                            #                             XX0=np.stack((X0,X1), axis=0)
                            #                             aa=np.array([a0,a1])
                            #                             sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,
                            #                                                 method='lm')
                            XX0=np.stack((X0,X1), axis=0)
                            aa=np.array([a0,a1])
                            sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,method='lm')

                            roots = sol.x
                            X2 = np.array([roots[0], roots[1], np.sqrt(roots[0] ** 2 + roots[1] ** 2 + 1)])
                            G.nodes[w]["Coords"] = X2.tolist()
                            #                    aa0=X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]
                            #                    aa1=X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]
                            #                    d = K * np.arccosh(-(X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]))
                            #                    d1= K * np.arccosh(-(X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]))
                            #                    print(aa0-a0, aa1-a1)

                            #                    print(tree[u][w]["distance"] - d, G[v][w]["distance"]-d1)

                            alpha = self.hyperbolicCosLaw(G[v][w]["distance"], G[u][v]["distance"],
                                                          G[u][w]["distance"], K)
                            G.nodes[u]["angle"] += alpha
                            if G.has_edge(u, w):
                                G[u][w]['dir'] = G.nodes[u]["angle"]
                            G.nodes[w]["angle"] = np.pi - G.nodes[u]["angle"]
                            break
                else:
                    a = np.cosh(G[father][w]["distance"] / K**2)
                    alpha = np.pi / G.degree[father]
                    X0 = G.nodes[father]["Coords"]
                    X = self.embed(X0, a, alpha)
                    G.nodes[w]["Coords"] = X.tolist()
                    G.nodes[w]["Coords"] = X.tolist()
                    #                        d=K*np.arccosh(-(X[0]*X0[0]+X[1]*X0[1]-X[2]*X0[2]))
                    #                        print(tree[father][w]["distance"]-d,)
                    G.nodes[father]["angle"] += alpha
                    G[father][w]['dir'] = G.nodes[father]["angle"]
                    G.nodes[w]["angle"] = np.pi - G.nodes[father]["angle"]
                    dcount += 1
                anchored.add(w)
                for t in G[w]:
                    if t not in anchored:
                        if t in list(G.neighbors(71)):
                            print("HERE")
                        #                    processQueue.append((t, tree.nodes[t]["father"], tree[t][w]['distance']))
                        processQueue.append((t, w, triangles[t]))

        cnt = 0
        coords = np.zeros((G.number_of_nodes(), 3))
        for v in G.nodes:
            coords[cnt, 0] = G.nodes[v]["Coords"][0]
            coords[cnt, 1] = G.nodes[v]["Coords"][1]
            coords[cnt, 2] = G.nodes[v]["Coords"][2]
            cnt += 1
        print("count=", count, "dcount=", dcount)
        return G, root, coords

    def greedyEmbedding4(self, G, C):
        #by triangulation
        warnings.filterwarnings("error")
        nx.set_node_attributes(G, np.inf, "dir")
        count = 0
        dcount = 0
        K = 1 / np.sqrt(C)
        adjDict={n:nbrdict for n, nbrdict in G.adjacency()}
        numTriangles = 0
        triangles={}
        anchored=set()
        for w in adjDict:
            triangles[w] = {}
            for u in adjDict[w]:
                triangles[w][u]=set()
        for w in adjDict:
            for u in adjDict[w]:
                for v in adjDict[u]:
                    if v in adjDict[w]:
                        if ( u not in triangles[w][v]) and (v not in triangles[w][u]):
                            triangles[w][u].add(v)
                            triangles[w][v].add(u)
                            numTriangles +=1
        sortedTriangle = dict(sorted(triangles.items(), key=lambda item: len(item[1]), reverse=True))
        root = next(iter(sortedTriangle))
        G.nodes[root]["Coords"]=[0,0,1]
        G.nodes[root]["angle"] = 0.0
        anchored.add(root)
        count = 0
        while numTriangles > 0:
            settled1 = False
            settled2 = False
            sortedTriangle = dict(sorted(triangles.items(), key=lambda item: len(item[1]), reverse=True))
            iterator=iter(sortedTriangle)
            for w in sortedTriangle:
                if w in anchored:
                    #one point is anchored
                    settled1 = True
                    settled2 = False
                    while (not settled2):
                        settled2 = False
                        sortedNeighbors = dict(sorted(triangles[w].items(), key=lambda item: len(item[1]), reverse=True))
                        for u in sortedNeighbors:
                            if u in anchored:
                                # a second point is anchored
                                settled2 = True
                                for v in sortedNeighbors[u]:
                                    # (w,u,v) can be triangularized
                                    if (v in anchored):
                                        #check if the distance are correct
                                        X0 = Hyperpoint(np.array(G.nodes[u]['Coords']), C)
                                        X1 = Hyperpoint(np.array(G.nodes[w]['Coords']), C)
                                        X2 = Hyperpoint(np.array(G.nodes[v]['Coords']), C)
                                        d1=X0.hdist(X1)
                                        d2=X0.hdist(X2)
                                        d3=X1.hdist(X2)
                                        print(d1-G[u][w]['distance'])
                                        print(d2-G[u][v]['distance'])
                                        print(d3-G[v][w]['distance'])
                                    else:
                                        count += 1
                                        X0 = G.nodes[u]["Coords"]
                                        X1 = G.nodes[w]["Coords"]
                                        a0 = -np.cosh(G[u][v]["distance"] / K ** 2)
                                        a1 = -np.cosh(G[w][v]["distance"] / K ** 2)
                                        #                             XX0=np.stack((X0,X1), axis=0)
                                        #                             aa=np.array([a0,a1])
                                        #                             sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,
                                        #                                                 method='lm')
                                        XX0 = np.stack((X0, X1), axis=0)
                                        aa = np.array([a0, a1])
                                        sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa),
                                                            jac=self.jacobian, method='lm')
                                        roots = sol.x
                                        X2 = np.array([roots[0], roots[1], np.sqrt(roots[0] ** 2 + roots[1] ** 2 + 1)])
                                        G.nodes[v]["Coords"] = X2.tolist()
                                        anchored.add(v)
                                numTriangles -= len(triangles[w][u])
                                triangles[w][u].clear()
                        if not settled2:
                            #we have to set one of the neighbors of w
                            uu=next(iter(sortedNeighbors))
                            a = np.cosh(G[uu][w]["distance"] / K ** 2)
                            alpha = np.pi / G.degree[w]
                            X0 = G.nodes[w]["Coords"]
                            X = self.embed(X0, a, alpha)
                            G.nodes[uu]["Coords"] = X.tolist()
                            G.nodes[uu]["Coords"] = X.tolist()
                            #                        d=K*np.arccosh(-(X[0]*X0[0]+X[1]*X0[1]-X[2]*X0[2]))
                            #                        print(tree[father][w]["distance"]-d,)
                            G.nodes[w]["angle"] += alpha
                            G[uu][w]['dir'] = G.nodes[w]["angle"]
                            G.nodes[uu]["angle"] = np.pi - G.nodes[w]["angle"]
                            anchored.add(uu)

            w=next(iterator)
            u=next(iter(sortedNeighbors))
            for v in sortedNeighbors[u]:
                print(w,u,v)
        triangles = nx.triangles(G)
        root = max(triangles, key=triangles.get)
        anchored = set()
        processQueue = []
        #        degree_sequence = sorted(G.degree(), key=lambda tup:tup[1], reverse=True)
        #        root=degree_sequence[0][0]
        w = root
        W = set(list(G.neighbors(w)))
        triangleDict = dict()
        triangleDict[w]=dict()
        for u in W:
            triangleDict[w][u] =[]
            U = set(list(G.neighbors(u)))
            UW = U & W
            for v in UW:
                triangleDict[w][u].append((u,v,w))
        G.nodes[root]["Coords"] = np.array([0, 0, 1])
        G.nodes[root]["angle"] = 0.0
        anchored.add(root)
        for u in G[root]:
            if len(triangleDict[u]) > 0:
                processQueue.append((u, w, counter[u] / 2))
        #            processQueue.append((u, root, triangles[u]))
        #            processQueue.append((u,root,G.degree[u]))
        while processQueue:
            processQueue.sort(key=lambda tup: tup[2], reverse=True)
            triangulation = False
            fullneigh = []
            w, father, d = processQueue.pop(0)  # w is the point to position
            if (w not in anchored):
                W = set(list(G.neighbors(w)))
                triangleDict= dict()
                counter = dict()
                for u in W:
                    counter[u] = 0
                    triangleDict[u] = []
                    # count triangles that are based on (u,w)
                    U = set(list(G.neighbors(u)))
                    UW = U & W
                    for v in UW:
                        if G.has_edge(u, v):  # (u,v,w) is a triangle
                            counter[u] += 1
                            if ((u in anchored) and (v in anchored)):
                                triangulation = True
                                triangleDict[u].append((u, v, w))
                counterList = sorted(counter, key=counter.get, reverse=True)
                if triangulation:  # we can position w by triangylation
                    for u in counterList:
                        if triangleDict[u]:
                            u = triangleDict[u][0][0]
                            v = triangleDict[u][0][1]
                            count += 1
                            X0 = G.nodes[u]["Coords"]
                            X1 = G.nodes[v]["Coords"]
                            a0 = -np.cosh(G[u][w]["distance"] / K**2)
                            a1 = -np.cosh(G[v][w]["distance"] / K**2)
                            #                             XX0=np.stack((X0,X1), axis=0)
                            #                             aa=np.array([a0,a1])
                            #                             sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,
                            #                                                 method='lm')
                            XX0=np.stack((X0,X1), axis=0)
                            aa=np.array([a0,a1])
                            sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,method='lm')

                            roots = sol.x
                            X2 = np.array([roots[0], roots[1], np.sqrt(roots[0] ** 2 + roots[1] ** 2 + 1)])
                            G.nodes[w]["Coords"] = X2.tolist()
                            #                    aa0=X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]
                            #                    aa1=X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]
                            #                    d = K * np.arccosh(-(X2[0] * X0[0] + X2[1] * X0[1] - X2[2] * X0[2]))
                            #                    d1= K * np.arccosh(-(X2[0] * X1[0] + X2[1] * X1[1] - X2[2] * X1[2]))
                            #                    print(aa0-a0, aa1-a1)

                            #                    print(tree[u][w]["distance"] - d, G[v][w]["distance"]-d1)

                            alpha = self.hyperbolicCosLaw(G[v][w]["distance"], G[u][v]["distance"],
                                                          G[u][w]["distance"], K)
                            G.nodes[u]["angle"] += alpha
                            if G.has_edge(u, w):
                                G[u][w]['dir'] = G.nodes[u]["angle"]
                            G.nodes[w]["angle"] = np.pi - G.nodes[u]["angle"]
                            break
                else:
                    a = np.cosh(G[father][w]["distance"] / K**2)
                    alpha = np.pi / G.degree[father]
                    X0 = G.nodes[father]["Coords"]
                    X = self.embed(X0, a, alpha)
                    G.nodes[w]["Coords"] = X.tolist()
                    G.nodes[w]["Coords"] = X.tolist()
                    #                        d=K*np.arccosh(-(X[0]*X0[0]+X[1]*X0[1]-X[2]*X0[2]))
                    #                        print(tree[father][w]["distance"]-d,)
                    G.nodes[father]["angle"] += alpha
                    G[father][w]['dir'] = G.nodes[father]["angle"]
                    G.nodes[w]["angle"] = np.pi - G.nodes[father]["angle"]
                    dcount += 1
                anchored.add(w)
                for t in G[w]:
                    if t not in anchored:
                        if t in list(G.neighbors(71)):
                            print("HERE")
                        #                    processQueue.append((t, tree.nodes[t]["father"], tree[t][w]['distance']))
                        processQueue.append((t, w, triangles[t]))

        cnt = 0
        coords = np.zeros((G.number_of_nodes(), 3))
        for v in G.nodes:
            coords[cnt, 0] = G.nodes[v]["Coords"][0]
            coords[cnt, 1] = G.nodes[v]["Coords"][1]
            coords[cnt, 2] = G.nodes[v]["Coords"][2]
            cnt += 1
        print("count=", count, "dcount=", dcount)
        return G, root, coords


    def fitMat(self,X0,adjMat,  incMat,C):
        #adjDict is a dictionary with 3 entries sources, targets, distances
        #fit to a given matrix DEL
        self.X0=X0
        aadjMat = triu(adjMat, k=1).tocoo()
        self.sources = list()
        self.targets = list()
        self.distances = list()
        k=0
        incMat1= csr_matrix((aadjMat.data, (aadjMat.row, aadjMat.col)), shape = incMat.shape)
        incMat2 = incMat1
        self.DELvect=aadjMat.data
        self.sources=aadjMat.row.tolist()
        self.targets=aadjMat.col.tolist()


#        for i, j, v in zip(aadjMat.row, aadjMat.col, aadjMat.data):
#            incMat1[i,k] = 1.0
#            incMat2[j,k] = 1.0
#            self.sources.append(i)
#            self.targets.append(j)
#            self.distances.append(v)
#            self.DELvect[k]=v
#            k += 1
        self.D0=self.hdistmat(X0[self.sources],X0[self.targets],C)
        E_1 = np.inf
        sM = 10
        r = 0.5
        p = 0.0
        t = 1
        searchitermax = 0
        X=X0
        while True:
        #    z = self.preprocess(z)
            E,D,gE = self.fgrad(X, incMat, C)
#            gE=-gE
            gn = np.max(np.sqrt(gE[:,0]**2+gE[:,1]**2))
            rM = 10

            if E < self.eE:
                if self.verb > 0:
                    print('exit: STRESS <eE')
                break
            elif E_1 - E < self.eEd:
                if self.verb > 0:
                    print('exit: DELTA E <eEd')
                    if E_1 < E:
                        print('STRESS GROWS')
                break
            elif gn < self.eE:
                if self.verb > 0:
                    print('exit: INF NORM OF GRADIENT <eE')
                break
            elif rM < self.ez:
                if self.verb > 0:
                    print('exit: rM <ez')
                break
            elif t > self.TM:
                if self.verb > 0:
                    print('exit: ITERATION LIMIT')
                break

            if self.verb > 0:
                print(t)
                print(E)
                print(E_1 - E)

            E_1 = E


#            M1t = np.array(self.M1(z, gE, 0).tolist(),dtype=np.complex128)
#            dq0 = np.sum(np.real(gE)*np.real(M1t)+np.imag(gE)*np.imag(M1t))
            dq0=self.dq0(self.sources,self.targets,X,gE,D,incMat1,incMat2)

            r, searchiter = self.bhlsearch(X, gE, E, dq0, 0.5, rM,C)

            searchitermax = max(searchitermax, searchiter)

            X = self.M(-r * gE, X)

            t += 1
        if self.verb > 0:
            print(f'{len(X)} points {t} iterations, E = {E:.12g}')
        self.X=X
        self.perf=E
        self.D=D
        return [X, E]

    def hdistmat(self, Xsources, Xtargets, C):
        #np.arccosh(-(X0[0] * X[0] + X0[1] * X[1] - X0[2] * X[2]))
        #Xsources (Nx2 matrix, [r, alpha]
        K=1/np.sqrt(C)
        Xs=Xsources
        Xt=Xtargets
        a=Xs[:,0]*Xt[:,0]+Xs[:,1]*Xt[:,1]-Xs[:,2]*Xt[:,2]
        a[np.where(abs(a)<1)]=-1-1e-6
        d=K**2*np.arccosh(-a)
        return d

    def fstress_rmse( self, D,Delta):
        errMat=D-Delta
        msqe=np.sum(errMat**2)
        rmse=np.sqrt(msqe/D.shape[0])
        return rmse, errMat

    def grad_rmse(self,Xsources,Xtargets,err,C,incMat):
        K=1/np.sqrt(C)
        Xs=Xsources
        Xt=Xtargets
        gE=np.zeros((incMat.shape[0],2))
        B=Xs[:,0]*Xt[:,0]+Xs[:,1]*Xt[:,1]-Xs[:,2]*Xt[:,2]
        B[np.where(B>=-1)]=-1-1e-6
        dExi=-2*K*err*(Xt[:,0]-Xs[:,0]*Xt[:,2]/Xs[:,2])/np.sqrt(B**2-1)
        dEyi=-2*K*err*(Xt[:,1]-Xs[:,1]*Xt[:,2]/Xs[:,2])/np.sqrt(B**2-1)
        Xs=Xtargets
        Xt=Xsources
        dExj=-2*K*err*(Xt[:,0]-Xs[:,0]*Xt[:,2]/Xs[:,2])/np.sqrt(B**2-1)
        dEyj=-2*K*err*(Xt[:,1]-Xs[:,1]*Xt[:,2]/Xs[:,2])/np.sqrt(B**2-1)
        gE[:,0]=incMat@(dExi+dExj)
        gE[:,1]=incMat@(dEyi+dEyj)
        return gE

    def fgrad_rmse(self,X,incMat, C):
        Xsources=X[self.sources,:]
        Xtargets=X[self.targets,:]
        D = self.hdistmat(Xsources,Xtargets,C)
        E,err= self.fstress_rmse(D,self.DELvect)
        gE=self.grad_rmse(Xsources, Xtargets, err,C, incMat)
        return E,D,gE

    def fstress_rrmse(self, D, Delta):
        errMat = np.abs(D-Delta)/Delta
        msqe = np.sum(errMat)
        return msqe, errMat

    def grad_rrmse(self, Xsources, Xtargets, err, C, incMat):
        K = 1 / np.sqrt(C)
        Xs = Xsources
        Xt = Xtargets
        gE = np.zeros((incMat.shape[0], 2))
        B = Xs[:, 0] * Xt[:, 0] + Xs[:, 1] * Xt[:, 1] - Xs[:, 2] * Xt[:, 2]
#        dExi = -2 * K * err * (Xt[:, 0] - Xs[:, 0] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
#        dEyi = -2 * K * err * (Xt[:, 1] - Xs[:, 1] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        dExi = -K* np.sign(err)*(Xt[:, 0] - Xs[:, 0] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        dEyi = -K * np.sign(err)*(Xt[:, 1] - Xs[:, 1] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        Xs = Xtargets
        Xt = Xsources
#        dExj = -2 * K * err * (Xt[:, 0] - Xs[:, 0] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
#        dEyj = -2 * K * err * (Xt[:, 1] - Xs[:, 1] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        dExj = -K *np.sign(err)* (Xt[:, 0] - Xs[:, 0] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        dEyj = -K* np.sign(err)*(Xt[:, 1] - Xs[:, 1] * Xt[:, 2] / Xs[:, 2]) / (self.DELvect*np.sqrt(B ** 2 - 1))
        gE[:, 0] = incMat @ (dExi + dExj)
        gE[:, 1] = incMat @ (dEyi + dEyj)
        return gE

    def fgrad_rrmse(self, X, incMat, C):
        Xsources = X[self.sources, :]
        Xtargets = X[self.targets, :]
        D = self.hdistmat(Xsources, Xtargets, C)
        E, err = self.fstress_rrmse(D, self.DELvect)
        gE = self.grad_rrmse(Xsources, Xtargets, err, C, incMat)
        return E, D, gE

    def fstress_sam(self,z,DELvect,sources,targets):
        [D,theta,denom] = self.hdistmat(z,sources,targets, C)
        np.seterr(divide='ignore', invalid='ignore')
        errMat = (D -  DEL)
        samErr=errMat**2/DEL
        np.fill_diagonal(samErr, 0)
        rerrsq=np.sum(samErr)
        return rerrsq, D, errMat, theta, denom

    def fstressgrad_sam(self,z, DEL, C):
        E, D, errMat, theta, denom = self.fstress_sam(z, DEL, C)
        thetac = theta.copy()
        # settle too close together points
        thetac[theta < 1e-9] = 1e-9
        # settle too far away points
        thetac[theta > 1 - 1e-12] = 1e-12
        realz = np.real(z)
        imagz = np.imag(z)
        coeff = errMat / (np.sqrt(C) * DEL*(1 - np.square(thetac)) * thetac * np.square(np.square(denom)))
        numeratorx = np.diag(2 * (np.abs(z)**2 - 1).ravel())@(
                    (realz ** 2 - imagz ** 2+1) @ realz.T + 2 * (realz * imagz) @ imagz.T - realz @ (
                        np.abs(z)**2 + 1).T )
        dExMat = coeff * numeratorx
        np.fill_diagonal(dExMat, 0)
        dExMat[dExMat > 1e6] = 1e6
        dExMat[dExMat < -1e6] = -1e6
        dEx = np.sum(dExMat, axis=0)
        numeratory = np.diag(2 * (np.abs(z)**2 - 1).ravel()) @ (
                    (imagz ** 2 - realz ** 2+1)@imagz.T + 2 * (realz * imagz) @ realz.T - imagz @ (
                        np.abs(z)**2 + 1).T + imagz)
        dEyMat = coeff * numeratory
        np.fill_diagonal(dEyMat, 0)
        dEyMat[dEyMat > 1e6] = 1e6
        dEyMat[dEyMat < -1e6] = -1e6
        dEy = np.sum(dEyMat, axis=0)
        gE = (dEx + 1j * dEy).reshape((len(z),1))
        return E, gE

    def preprocess(self,z):
        I = np.where(np.abs(z) > 1)
        z[I] = z[I] / np.abs(z[I])
        I = np.where(np.abs(np.log(np.abs(z))) < 1e-5)
        z[I] = z[I] * (1 - 1e-4)
        return z

    def M(self,r, X):
        Xr=np.zeros(X.shape)
        Xr[:,0]=X[:,0]+r[:,0]
        Xr[:,1]=X[:,1]+r[:,1]
        Xr[:,2]=np.sqrt(Xr[:,0]**2+Xr[:,1]**2+1)
        return Xr

    def dq0(self, sources, targets,X,g, D,incMat1, incMat2):
        err=D-self.DELvect
        gs=g[sources,:]
        gt=g[targets,:]
        Xs=X[sources,:]
        Xt=X[targets,:]
        B=Xs[:,0]*Xt[:,0]+Xs[:,1]*Xt[:,1]-Xs[:,2]*Xt[:,2]
        As=Xs[:,0]*gs[:,0]+Xs[:,1]*gs[:,1]
        At=Xt[:,0]*gt[:,0]+Xt[:,1]*gt[:,1]
        B[np.where(B >= -1)] = -1 - 1e-6
        dq0=(Xs[:,0]*gt[:,0]+Xt[:,0]*gs[:,0]+Xs[:,1]*gt[:,1]+Xt[:,1]*gs[:,1]-As*Xt[:,2]/Xs[:,2]-At*Xs[:,2]/Xt[:,2])/np.sqrt(B**2-1)
        return np.sum(dq0)

    def bhlsearch(self, X0,gE, q0, dq0, p, rM, C):
        pdq0 = p * dq0
        iter = 0
        r=rM
        Xr=self.M(-r * gE, X0)
        D = self.hdistmat(Xr[self.sources],Xr[self.targets],C)
        qr,err=self.fstress(D,self.DELvect)
        while (r < rM) and (qr < q0 + pdq0 * r):
            r = max(2 * r,rM)
            zr = self.M(-r * gE, X0)
            D = self.hdistmat(zr[self.sources], zr[self.targets], C)
            qr,err = self.fstress(D, self.DELvect)
            iter +=  1
        while ((r >= rM) or (qr > q0 + pdq0 * r)) and iter<1000:
            r = r / 2
            Xr = self.M(-r * gE, X0)
            D = self.hdistmat(Xr[self.sources], Xr[self.targets], C)
            qr,err = self.fstress(D, self.DELvect)
#            print(qr,q0+pdq0 * r)
            iter +=1
        return r,iter
    def toPD(self):
        self.Z=self.X[:,0]/(self.X[:,2]+1)+1j*self.X[:,1]/(self.X[:,2]+1)

    def PD_translation(self,c):
        self.Z = (self.Z + c) / (np.conj(c) * self.Z + 1)
        return self.Z


class CombinatorialEmbedding:
    def __init__(self, nodeSet, fullMetricGraph, C, dim, verb, origin=None, dist0=None, stress='rmse'):
        self.verb = verb
        self.MetricGraph = fullMetricGraph
        self.fullGraph = fullMetricGraph.G
        self.G = nx.subgraph(self.fullGraph, nodeSet).copy()
        self.num = self.G.number_of_nodes()
        self.C = C
        self.K = 1 / np.sqrt(C)
        self.dim = dim
        self.anchors = []
        self.anchorsCoords = []
        self.anchorEdges = []
        self.anchorEdge=[]
        self.nonAnchorEdge = []
        self.X0 = np.zeros(self.dim * self.num)
        self.nodes = []
        self.edges = []
        self.DELvect = np.zeros(self.G.number_of_edges())
        self.translate = False
        self.count=0
        self.dcount=0
        for n in self.G.nodes():
            self.nodes.append(n)
            if 'Coords' in self.G.nodes[n]:
                self.anchors.append(n)
                self.anchorsCoords.append(self.G.nodes[n]['Coords'])
        ind = 0
        for u,v in self.G.edges():
            if 'Coords' in self.G.nodes[u] and 'Coords' in self.G.nodes[v]:
                self.anchorEdges.append((u,v))
        self.dist0 = np.zeros(self.G.number_of_nodes())
        self.sources = []
        self.targets = []
        oorigin = np.zeros(self.dim + 1)
        oorigin[-1] = 1
        if self.anchors:
            if len(self.anchors) < self.G.number_of_nodes():
                # we need to fit
                self.toFit = True
                anchorsCoords = np.array(self.anchorsCoords)
                if origin is None:
                    self.origin = self.anchors[0]
                else:
                    if origin in self.anchors:
                        self.origin = origin
                    else:
                        self.origin = self.anchors[0]
                if not (np.array_equal(origin, self.anchorsCoords[0])):
                    # we need to translate the choosen vertex to the origin
                    self.translate = True
                    self.PP = anchorsCoords[0]
#                        anchorsCoords = self.translationToOrigin(anchorsCoords, np.array(anchorsCoords[0]))
                    #                    ind +=1
            else:
                # all vertices are already anchored
                # we have nothing to do
                ind = 0
                for u, v, data in self.G.edges(data=True):
                    self.edges.append((u, v))
                    self.DELvect[ind] = data['distance']
                    if (u in self.anchors) and (v in self.anchors):
                        self.anchorEdge.append(ind)
                    else:
                        self.nonAnchorEdge.append(ind)
                    ind += 1
                self.toFit = False
        else:
            self.toFit = True
            if oorigin is None:
                self.origin = list(self.G.nodes)[0]
            else:
                self.origin = list(self.G.nodes)[0]
                oorigin = np.zeros(self.dim + 1)
                oorigin[-1] = 1

            self.fullGraph.nodes[self.origin]['Coords'] = oorigin
            self.G.nodes[self.origin]['Coords'] = oorigin
            self.anchors.append(self.origin)
            self.translate = False
        if self.toFit:
            ind = 0
            for u, v, data in self.G.edges(data=True):
                self.edges.append((u, v))
                self.DELvect[ind] = data['distance']
                if (u in self.anchors) and (v in self.anchors):
                    self.anchorEdge.append(ind)
                else:
                    self.nonAnchorEdge.append(ind)
                ind += 1
        for u, v, data in self.G.edges(data=True):
            self.edges.append((u, v))
            I = self.nodes.index(u)
            J = self.nodes.index(v)
            self.sources.append(I)
            self.targets.append(J)

    def greedyEmbedding3(self):
        G=self.G
        C=self.C
        warnings.filterwarnings("error")
        nx.set_node_attributes(G, np.inf, "dir")
        processQueue = []
        triangles = nx.triangles(G)
        visited=set()
        anchoredEdges=set()
        if self.anchors:
            anchored = set(self.anchors)  # set of nodes that have been anchored
            for v in anchored:
                if 'angle' not in G.nodes[v]:
                    G.nodes[v]['angle'] =0.0
        else:
            root = max(triangles, key=triangles.get)
            oorigin = np.zeros(self.dim + 1)
            oorigin[-1] = 1
            self.G.nodes[root]['Coords'] = oorigin
            self.G.nodes[root]['angle'] = 0.0
            anchored = set(root)
        for (u,v) in self.anchorEdges:
            anchoredEdges.add((u,v))
            processQueue.append((u, v, 1e6 + max(triangles[u], triangles[v]),'e'))
        for w in anchored:
            W = set(list(G.neighbors(w))) #neighbors of an anchored point.
            for u in W:
                if u not in anchored:  # both u and w are anchored so all triangle with base uw can be settled
                    processQueue.append((u,w,max(triangles[u],triangles[w]),'v')) # 1e6 is used to be sure that triangulated node are calculated first
            visited.add(w)
        unmatched =0
        while processQueue:
            settled = False
            processQueue.sort(key=lambda tup: tup[2], reverse=True)
#            triangulation = False
            w, father, d, type= processQueue.pop(0) # w is the point to position and father is supposed to be already positionned
            u=father
            if type == 'e': #this an edge basis
                W = set(list(G.neighbors(w)))
                U = set(list(G.neighbors(father)))
                WU = W & U;
                for v in W:
                    if v in WU:
                        if v not in anchored:
                            self.triangulate(v, father, w)# we can triangulate v by father, and w
                            anchored.add(v)
                            anchoredEdges.add((u, v))
                            self.G[u][v]['edist'] = \
                            self.hdistmat(np.array(self.G.nodes[u]['Coords'])[:, np.newaxis].transpose(),
                                          np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                          self.K)[0]
                            self.fullGraph[u][v]['edist'] = self.G[u][v]['edist']
                            self.G[u][v]['type'] = 'e'
                            self.fullGraph[u][v]['type'] = 'e'
                            self.G[w][v]['edist'] = \
                            self.hdistmat(np.array(self.G.nodes[w]['Coords'])[:, np.newaxis].transpose(),
                                          np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                          self.K)[0]
                            self.fullGraph[w][v]['edist'] = self.G[w][v]['edist']
                            self.G[w][v]['type'] = 'e'
                            self.fullGraph[w][v]['type'] = 'e'
                            self.count+=2
                            processQueue.append((v, u, 1e6 +max(triangles[u], triangles[v]), 'e'))  # 1e6 is used to be sure that triangulated node are calculated first
                            anchoredEdges.add((u, v))
                            processQueue.append((v, w, 1e6 +max(triangles[w], triangles[v]), 'e'))  # 1e6 is used to be sure that triangulated node are calculated first
                            anchoredEdges.add((w, v))
                        else:
                            if (u,v) not in anchoredEdges and (v,u) not in anchoredEdges:
                                self.G[u][v]['edist'] = self.hdistmat(np.array(self.G.nodes[u]['Coords'])[:, np.newaxis].transpose(),
                                                                           np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                                                           self.K)[0]
                                self.fullGraph[u][v]['edist'] = self.G[u][v]['edist']
                                self.G[u][v]['type'] = 'u'
                                self.fullGraph[u][v]['type'] = 'u'
                                unmatched += 1
                            if (w,v) not in anchoredEdges and (v,w) not in anchoredEdges:
                                self.G[w][v]['edist'] = self.hdistmat(np.array(self.G.nodes[w]['Coords'])[:, np.newaxis].transpose(),
                                                                           np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                                                           self.K)[0]
                                self.fullGraph[w][v]['edist'] = self.G[w][v]['edist']
                                self.G[w][v]['type'] = 'u'
                                self.fullGraph[w][v]['type'] = 'u'
                                unmatched += 1
                    else:
                        if v not in anchored:
                            if v != father:
                                processQueue.append((v, w, max(triangles[w], triangles[v]), 'v'))
                        else:
                            if v in W and 'edist' not in self.G[w][v]:
                                self.G[w][v]['edist'] = \
                                self.hdistmat(np.array(self.G.nodes[w]['Coords'])[:, np.newaxis].transpose(),
                                              np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                              self.K)[0]
                                self.fullGraph[w][v]['edist'] = self.G[w][v]['edist']
                                self.G[w][v]['type'] = 'u'
                                self.fullGraph[w][v]['type'] = 'u'
                                unmatched += 1
                            if v in U and 'edist' not in self.G[u][v] :
                                self.G[u][v]['edist'] = \
                                self.hdistmat(np.array(self.G.nodes[u]['Coords'])[:, np.newaxis].transpose(),
                                              np.array(self.G.nodes[v]['Coords'])[:, np.newaxis].transpose(),
                                              self.K)[0]
                                self.fullGraph[u][v]['edist'] = self.G[u][v]['edist']
                                self.G[u][v]['type'] = 'u'
                                self.fullGraph[u][v]['type'] = 'u'
                                unmatched += 1


            if type == 'v':
                if w not in anchored:
                    self.linearPosition(w, father)
                    anchored.add(w)
                    anchoredEdges.add((u, w))
                    processQueue.append((u, w, 1e6 + max(triangles[u], triangles[v]),
                                     'e'))  # 1e6 is used to be sure that triangulated node are calculated first
                    self.dcount += 1
                    self.G[w][father]['edist'] = self.hdistmat(np.array(self.G.nodes[w]['Coords'])[:, np.newaxis].transpose(),
                                              np.array(self.G.nodes[father]['Coords'])[:, np.newaxis].transpose(),
                                              self.K)[0]
                    self.fullGraph[w][father]['edist'] = self.G[w][father]['edist']
                    self.G[w][father]['type'] = 'v'
                    self.fullGraph[w][father]['type'] = 'v'
                    W = set(list(G.neighbors(w)))
                    for v in W:
                        if v not in anchored and v != father:
                            processQueue.append((v,w, max(triangles[w], triangles[v]), 'v'))
                else:
                    if 'edist' not in self.G[w][father]:
                        self.G[w][father]['edist'] = \
                        self.hdistmat(np.array(self.G.nodes[w]['Coords'])[:, np.newaxis].transpose(),
                                      np.array(self.G.nodes[father]['Coords'])[:, np.newaxis].transpose(),
                                      self.K)[0]
                        self.fullGraph[w][father]['edist'] = self.G[w][father]['edist']
                        self.G[w][father]['type'] = 'v'
                        self.fullGraph[w][father]['type'] = 'v'
                        unmatched += 1

            visited.add(w)
        self.coords = np.zeros((G.number_of_nodes(), 3))
        cnt=0
        for v in G.nodes:
            self.coords[cnt, 0] = G.nodes[v]["Coords"][0]
            self.coords[cnt, 1] = G.nodes[v]["Coords"][1]
            self.coords[cnt, 2] = G.nodes[v]["Coords"][2]
            self.fullGraph.nodes[v]["Coords"] = G.nodes[v]["Coords"]
            self.fullGraph.nodes[v]["angle"] = G.nodes[v]["angle"]
            cnt += 1
        Xsources = self.coords[self.sources, :]
        Xtargets = self.coords[self.targets, :]
        self.D = self.hdistmat(Xsources, Xtargets,self.K)
        self.delta = self.D-self.DELvect
        self.sserror = np.sum(self.delta**2)
        if self.sserror > 1:
            pass
        if self.G.number_of_edges()>0:
            self.rmse = np.sqrt(self.sserror/self.G.number_of_edges())
        else:
            self.rmse = 0
        self.rerror=np.sum((self.delta/self.DELvect)**2)
        print("count=", self.count, "dcount=", self.dcount, "unmatched=", G.number_of_edges()-len(anchoredEdges))
        return self.fullGraph, self.coords

    def embed(self, X0, a, alpha):
        X0=mp.matrix(X0).transpose()
        r=mp.sqrt(a**2-1)
        mp.matrix(3)
        X=r*mp.cos(alpha)
        Y=r*mp.sin(alpha)
        Z=a
        XX=mp.matrix([[X,Y,Z]])
        if (mp.almosteq(X0[2],1)):
            if mp.fabs(XX[0] ** 2 + XX[1] ** 2 - XX[2] ** 2 + 1) > 1e-1:
                pass
            return(np.array(XX.tolist(),dtype=np.float64).ravel())
        else:
            u=X0
            u[2] -=1
            u=u/mp.sqrt(u[0]**2+u[1]**2-u[2]**2)
            X1=XX-2*(X*u[0]+Y*u[1]-Z*u[2])*u
            X1 = X1.apply(mp.re)
            return (np.array(X1.tolist(), dtype=np.float64).ravel())

    def hdistmat(self, Xsources, Xtargets, K):
        #np.arccosh(-(X0[0] * X[0] + X0[1] * X[1] - X0[2] * X[2]))
        #Xsources (Nx2 matrix, [r, alpha]
        Xs=Xsources
        Xt=Xtargets
        a=Xs[:,0]*Xt[:,0]+Xs[:,1]*Xt[:,1]-Xs[:,2]*Xt[:,2]
        a[np.where(abs(a)<1)]=-1-1e-6
        d=K**2*np.arccosh(-a)
        return d

    def equations(self, X, XX0, a):#XX is a matrix of k points in k dimension and a an array of k values
        XX=np.append(X,-np.sqrt(np.sum(X*X)+1))
        aa=np.matmul(XX0, XX.transpose())
        F= aa-a
        return F

    def jacobian(self, X,XX0,a):
        XX = X/np.sqrt(np.sum(X * X) + 1)
        PP=np.outer(XX0[:,-1], XX)
        prime=XX0[:,0:-1]-PP
#        prime[0,0]=X0[0]-X0[2]*X2[0]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[0,1]=X0[1]-X0[2]*X2[1]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[1,0]=X1[0]-X1[2]*X2[0]/np.sqrt(X2[0]**2+X2[1]**2+1)
#        prime[1,1]=X1[1]-X1[2]*X2[1]/np.sqrt(X2[0]**2+X2[1]**2+1)
        return prime

    def hyperbolicCosLaw(self, a,b,c, K):
        if self.trianglecheck(a,b,c):
            if (c+b==a):
                alpha = np.pi
            else:
                if (np.cosh(b/K)*np.cosh(c/K)-np.cosh(a/K))/(np.sinh(b/K)*np.sinh(c/K))>1.0:
                    alpha = 0.0
                else:
                    alpha=np.arccos((np.cosh(b/K)*np.cosh(c/K)-np.cosh(a/K))/(np.sinh(b/K)*np.sinh(c/K)))
            return alpha
        else:
            if (c+b<a):
                return np.pi
            else :
                return 0.0

    def trianglecheck(self, a, b, c):
        if (a+b>=c) and (a+c>=b) and (b+c>=a):
            return True
        else:
            return False

    def triangulate(self, w, u, v): #triangulate w using u and v
        X0 = self.G.nodes[u]["Coords"]
        X1 = self.G.nodes[v]["Coords"]
        a0 = -np.cosh(self.G[u][w]["distance"] / self.K ** 2)
        a1 = -np.cosh(self.G[v][w]["distance"] / self.K ** 2)
        #                             XX0=np.stack((X0,X1), axis=0)
        #                             aa=np.array([a0,a1])
        #                             sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian,
        #                                                 method='lm')
        XX0 = np.stack((X0, X1), axis=0)
        aa = np.array([a0, a1])
        sol = optimize.root(self.equations, np.array([X0[0], X1[1]]), (XX0, aa), jac=self.jacobian, method='lm')

        roots = sol.x
        X2 = np.array([roots[0], roots[1], np.sqrt(roots[0] ** 2 + roots[1] ** 2 + 1)])
        self.G.nodes[w]["Coords"] = X2.tolist()
        alpha = self.hyperbolicCosLaw(self.G[v][w]["distance"], self.G[u][v]["distance"],
                                      self.G[u][w]["distance"], self.K)
        self.G.nodes[u]["angle"] += alpha
        if self.G.has_edge(u, w):
            self.G[u][w]['dir'] = self.G.nodes[u]["angle"]
        self.G.nodes[w]["angle"] = np.pi - self.G.nodes[u]["angle"]
        D=self.hdistmat(XX0,np.stack((X2,X2)),self.K)
        self.G[w][u]['edist']=D[0]
        self.fullGraph[w][u]['edist']=D[0]
        self.G[w][u]['type']='e'
        self.fullGraph[w][u]['type']='e'
        self.G[w][v]['edist']=D[1]
        self.fullGraph[w][v]['edist']=D[1]
        self.G[w][v]['type']='e'
        self.fullGraph[w][u]['type']='e'


    def linearPosition(self,w,father):
        a = np.cosh(self.G[father][w]["distance"] / self.K ** 2)
        alpha = np.pi / self.G.degree[father]
        X0 = self.G.nodes[father]["Coords"]
        X = self.embed(X0, a, alpha)
        self.G.nodes[w]["Coords"] = X.tolist()
        self.G.nodes[father]["angle"] += alpha
        self.G[father][w]['dir'] = self.G.nodes[father]["angle"]
        self.G.nodes[w]["angle"] = np.pi - self.G.nodes[father]["angle"]



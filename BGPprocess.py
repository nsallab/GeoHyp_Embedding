import math

import matplotlib.pyplot as plt
from networkx import convert_node_labels_to_integers

from HyperboloidMDS import HyperMDS
from scipy.io import savemat
import matlab.engine

import networkx as nx
#from graph_tool.all import *
from networkx import read_graphml, minimum_spanning_tree

import TreeProcessing as tp
import pickle as pkl
import numpy as np
from  MDS import MDS, GraphMDS
from scipy.sparse import csr_matrix

def getTreeDist(G,graphTree):
    treeDist = G.new_edge_property("double")
    treeDist.a = -1.0
    for e in G.edges():
        if treeDist[e]<0:
            neighbors=[]
            for neighbor in list(e.source().out_neighbors()):
                if treeDist[G.edge(e.source(),neighbor)]<0:
                    neighbors.append(neighbor)
            dist=shortest_distance(graphTree,e.source(),neighbors,weights=graphTree.edge_properties['distance'])
            for i in range(0,len(neighbors)):
                treeDist[G.edge(e.source(),neighbors[i])]=dist[i]
            neighbors=[]
            for neighbor in list(e.target().out_neighbors()):
                if treeDist[G.edge(e.target(),neighbor)]<0:
                    neighbors.append(neighbor)
            dist=shortest_distance(graphTree,e.target(),neighbors,weights=graphTree.edge_properties['distance'])
            for i in range(0,len(neighbors)):
                treeDist[G.edge(e.target(),neighbors[i])]=dist[i]
    return(treeDist)
path='./'

filename=path+"dist1199167200.pkl"
filehandler = open(filename,"rb")
dist1199167200=pkl.load(filehandler)
#filename=path+"distTree1199167200.pkl"
#filehandler = open(filename,"rb")
#distTree1199167200=pkl.load(filehandler)
#[rmse,rmsre,sams]=tp.getDistorsions2(dist1199167200,distTree1199167200)

def filter_edge(n1, n2):
    return (G[n1][n2]["tree"]==1)

G=nx.read_graphml(path+'treeprocessed_graphdumps1199167200.1199170800.graphml')
G=convert_node_labels_to_integers(G)

#rng = np.random.default_rng()
# Generate one random float uniformly distributed over the range [0, 1)
#randtheta=2*math.pi*rng.random((G.num_vertices(),1))
#randradius=rng.random((G.num_vertices(),1))
#zrand=(randradius*np.cos(randtheta)+1j*randradius*np.sin(randtheta)).ravel()
adjMat=nx.adjacency_matrix(G,weight="distance")
incMat=nx.incidence_matrix(G)
print("SPIELMAN tree process")

treesp = nx.subgraph_view(G, filter_edge=filter_edge)
#astretch,avstretchsq, aerrsq, gdist, sptdist, spstretch,sperr=getDistorsion(G,tree)
#print("Spielman:",astretch,avstretchsq,aerrsq)
sources = list()
targets = list()
distances = list()
k = 0
for i, j, v in G.edges(data=True):
    sources.append(i)
    targets.append(j)
    distances.append(v)

Tlist,levels,roots=tp.setDepth([treesp])
inttree=tp.distFilt(Tlist[0],0.0)
hyperMDS_SP = HyperMDS(1e-6,1e-7,1e-80,100,1,stress='rrmse')
treeGraph_SP,coords=hyperMDS_SP.sarkar_tree(treesp, 767, 0.61)
hyperMDS_SP.fitMat(coords,adjMat,  incMat,0.61)
hyperMDS_SP.toPD()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=hyperMDS_SP.DELvect, y=hyperMDS_SP.D0-hyperMDS_SP.DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error before MDS")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=hyperMDS_SP.DELvect, y=hyperMDS_SP.D-hyperMDS_SP.DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error after MDS")
plt.show()

eng = matlab.engine.start_matlab("-desktop")
eng.cd(r'./MDS-PD-matlab', nargout=0)
eng.workspace['xsp'] = eng.double(coords[:,0].ravel())
eng.workspace['ysp'] = eng.double(coords[:,1].ravel())
eng.workspace['zsp'] = eng.double(coords[:,2].ravel())
eng.workspace['xpsp'] = eng.double(hyperMDS_SP.X[:,0].ravel())
eng.workspace['ypsp'] = eng.double(hyperMDS_SP.X[:,1].ravel())
eng.workspace['zpsp'] = eng.double(hyperMDS_SP.X[:,2].ravel())
eng.workspace['Zxpdsp'] = eng.double(np.real(hyperMDS_SP.Z).ravel())
eng.workspace['Zypdsp'] = eng.double(np.imag(hyperMDS_SP.Z).ravel())

eng.hyperboloidPlot(eng.double(coords[:,0].ravel()),eng.double(coords[:,1].ravel()),eng.double(coords[:,2].ravel()),nargout=0)
eng.hyperboloidPlot(eng.double(hyperMDS_SP.X[:,0].ravel()),eng.double(hyperMDS_SP.X[:,1].ravel()),eng.double(hyperMDS_SP.X[:,2].ravel()),nargout=0)
eng.drawPD(eng.workspace['Zxpdsp'],eng.workspace['Zypdsp'], nargout=0)



graphMDS=GraphMDS(1e-6,1e-7,1e-18,100,1)
graphMDS.fitMat(zrand,adjMat,  incMat,0.63)
#mds=MDS(1e-6,1e-7,1e-18,100,1)
#mds.fitMat(zrand,dist1199167200,0.63)

mintree_map=min_spanning_tree(G, weights=G.edge_properties['distance'])
maxMap = G.new_edge_property("double")
maxMap.a = -G.edge_properties['distance'].a
maxtree_map=min_spanning_tree(G, weights=maxMap)

mingraphTree = GraphView(G,efilt=mintree_map)
maxgraphTree = GraphView(G,efilt=maxtree_map)
mintreeDist=getTreeDist(G,mingraphTree)
maxtreeDist=getTreeDist(G,maxgraphTree)

[minerrVect,minrerrVect, minrmse,minrmsre,minsams]=tp.getDistorsions1(G.edge_properties['distance'].a,mintreeDist.a)
[maxerrVect,maxrerrVect, maxrmse,maxrmsre,maxsams]=tp.getDistorsions1(G.edge_properties['distance'].a,maxtreeDist.a)

filename=path+"dist1199167200.pkl"
filehandler = open(filename,"rb")
dist1199167200=pkl.load(filehandler)

#[s1,s2,s3]=getDistorsions2(dist,distTree)≤

rng = np.random.default_rng()
# Generate one random float uniformly distributed over the range [0, 1)
randtheta=2*math.pi*rng.random((G.num_vertices(),1))
randradius=rng.random((G.num_vertices(),1))
zrand=randradius*np.cos(randtheta)+1j*randradius*np.sin(randtheta)
mds=MDS(1e-6,1e-7,1e-18,100,1)
mds.fitMat(zrand,dist1199167200,0.63)
import matplotlib.pyplot as plt

# Créer un scatter plot entre la surface habitable au-dessus du sol (Gr Liv Area) et le prix de vente (SalePrice)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=G.edge_properties['distance'].a, y=minerrVect)
plt.xlabel("Distance after Ricci")
plt.ylabel("Tree based Error")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=G.edge_properties['distance'].a, y=minrerrVect)
plt.xlabel("Distance after Ricci")
plt.ylabel("Tree based Error")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=G.edge_properties['distance'].a, y=maxerrVect)
plt.xlabel("Distance after Ricci")
plt.ylabel("Tree based Error")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=G.edge_properties['distance'].a, y=maxrerrVect)
plt.xlabel("Distance after Ricci")
plt.ylabel("Tree based Error")
plt.show()




treeG=minimum_spanning_tree(G,weight='distance')

G=load_graph(path+'processed_graphdumps1388556000.1388559600.graphml')
tree_map=min_spanning_tree(G, weights=G.edge_properties['distance'])
graphTree = GraphView(G,efilt=tree_map)
distTreeVect=shortest_distance(graphTree,weights=G.edge_properties['distance'])
distTree=np.zeros((G.num_vertices(),G.num_vertices()))
for i in range(0,G.num_vertices()):
        distTree[i]=distTreeVect[i]
filename=path+"distTree1388556000.pkl"
filehandler = open(filename,"wb")
pkl.dump(distTree,filehandler)
filehandler.close()

G=load_graph(path+'processed_graphdumps1527832800.1527836400.graphml')
tree_map=min_spanning_tree(G, weights=G.edge_properties['distance'])
graphTree = GraphView(G,efilt=tree_map)
distTreeVect=shortest_distance(graphTree,weights=G.edge_properties['distance'])
distTree=np.zeros((G.num_vertices(),G.num_vertices()))
for i in range(0,G.num_vertices()):
        distTree[i]=distTreeVect[i]
filename=path+"distTree1527832800.pkl"
filehandler = open(filename,"wb")
pkl.dump(distTree,filehandler)
filehandler.close()

G=load_graph(path+'processed_graphdumps1702188000.1702191600.graphml')
tree_map=min_spanning_tree(G, weights=G.edge_properties['distance'])
graphTree = GraphView(G,efilt=tree_map)
distTreeVect=shortest_distance(graphTree,weights=G.edge_properties['distance'])
distTree=np.zeros((G.num_vertices(),G.num_vertices()))
for i in range(0,G.num_vertices()):
        distTree[i]=distTreeVect[i]
filename=path+"distTree1702188000.pkl"
filehandler = open(filename,"wb")
pkl.dump(distTree,filehandler)
filehandler.close()



path='./'
G=read_graphml(path+'processed_graphdumps1199167200.1199170800.graphml')
filename=path+"dist1199167200.pkl"
filehandler = open(filename,"rb")
dist=pkl.load(filehandler)

TGlist=tp.distFilt(G,0)

Tlist=tp.minimumSpanningTree(TGlist)
[Tlist,depthT,roots]=tp.setDepth(Tlist)
distTreeList = tp.getTreeDistMat(Tlist)
dists=tp.getDistorsions2(G,Tlist,dist,distTreeList)


embedding = tp.sarkar_embedding(Tlist[0], roots[0], tau=0.5, weighted=True)
fembed=np.zeros((len(embedding),2))
for i in range(len(embedding)):
    fembed[i,0]=eval(str(embedding[i,0]))
    fembed[i,1]=eval(str(embedding[i,1]))
print('done')
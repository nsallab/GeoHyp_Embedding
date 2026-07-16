import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import HyperboloidMDS as hm
import copy
import time
# Path to the CSV file

utils = hm.Utils()
csv_file = '/Users/kave/Downloads/ca-CSphd.csv'
df = pd.read_csv(csv_file)
df = pd.read_csv(csv_file, sep=' ', header=None, names=['id1', 'id2'])
G = nx.Graph()

# Add edges from the DataFrame
for _, row in df.iterrows():
    G.add_edge(row['id1'], row['id2'], weight=1.0, distance=1.0)
nx.write_graphml(G,"CSPhd.graphml")
# Extract connected components
connected_components = [c for c in sorted(nx.connected_components(G), key=len)]
gg=nx.subgraph(G, connected_components[0]).copy()
gg=nx.convert_node_labels_to_integers(gg)
g=copy.deepcopy(gg)
N=g.number_of_nodes()
l=dict(nx.all_pairs_dijkstra_path_length(g,weight='distance', backend="parallel"))
cnt1=0
cnt2=0
distMat=np.zeros((N,N))
for node1 in g.nodes():
    cnt2=0
    for node2 in g.nodes():
        distMat[cnt1,cnt2]=l[node1][node2]
        cnt2 +=1
    cnt1 +=1
K=1
C=1/np.sqrt(K)
dim=2
mat=np.ones((N,N))*np.cosh(distMat[0,:]/K)
fitDist=2*np.cosh(distMat/K)+(mat-mat.transpose())**2-2
mat = np.ones((N,N))*fitDist[0,:]
D=-1/2*(fitDist-mat-mat.transpose())
#check how many eigenValues are >0
#check what is the rank of the EMD matrix
[l,U]=np.linalg.eigh(D)
l=np.real(l)
U=np.real(U)
#    print("Eigenvalues:",l)
#print("rank:",np.linalg.matrix_rank(fitDist))
I=np.argsort(l)
select=I[-dim+1:]
Xfit=np.zeros((N,dim))

fit=np.matmul(np.hstack((np.diag(np.sqrt(l[-dim+1:])),np.zeros((dim-1,N-dim+1)))),U.transpose())
Xfit[:,0:-1]=fit.transpose()
Xfit[:,-1]=np.sqrt(np.sum(Xfit[:,0:-1]**2,axis=1)+1)
estDistMat = np.zeros((N, N))
cnt=0
dist=np.zeros(g.number_of_edges())
estDist=np.zeros(g.number_of_edges())
for u,v, data in g.edges(data=True):
    X=hm.Hyperpoint(Xfit[u, :],C)
    Y=hm.Hyperpoint(Xfit[v, :],C)
    dist[cnt]=data['distance']
    estDist[cnt]=X.hdist(Y)
    cnt +=1
sum=np.sum((estDist-dist)**2)
print("Distance Error 1:", sum)
print("RMSE clique 1:", np.sqrt(sum/g.number_of_edges()))



g=copy.deepcopy(gg)
verticesSet=list(g.nodes())
MG=hm.MetricGraph(g)
hyper=hm.HyperHyp(verticesSet,MG,1,5,1,origin=0)
print("Hyperboloid gradient")
start_time = time.time()
[coords,K]=hyper.HYP_fit()
print("Execution time: ", time.time()-start_time)
print("Hyperboloid gradient rmse:", hyper.rmse)
print("Curvature:",1/np.sqrt(K))
print("Hyperboloid MAP:", utils.MAP(MG.G,coords))
fig, ax = plt.subplots(figsize=(10, 6))
err=(np.sum(hyper.DELvect)/np.sum(hyper.D))*hyper.D-hyper.DELvect
ax.scatter(x=hyper.DELvect, y=err)
sserr=np.sum(((np.sum(hyper.DELvect)/np.sum(hyper.D))*hyper.D - hyper.DELvect)**2)
print("SSErr:", sserr)
print("RMSE:", np.sqrt(sserr/MGG.G.number_of_edges()))
plt.xlabel("Target Distance")
plt.ylabel("error after MDS Hyperbolic")
plt.show()
del hyper

MGG = copy.deepcopy(MG)
verticesSet=list(MGG.G.nodes())
C=1/np.sqrt(K)
hyper=hm.HyperMDS(verticesSet, MGG, 1 , 5,1 )
print("Euclidean gradient")
start_time = time.time()
coords=hyper.MDS_fit()
print("Execution time: ", time.time() - start_time)
print("Euclidean Gradient rmse:", hyper.rmse)
print("Euclidean MAP:", utils.MAP(MGG.G,coords))
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=hyper.DELvect, y=hyper.D - hyper.DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error after MDS Hyperbolic")
plt.show()

del hyper



#Validation 4: Calculate CliqueDecomposition and Embedding
MGG=copy.deepcopy(MG)
print("clique decomposition")
start_time = time.time()
hyper=hm.HypMDSEmbedding(MGG,1)
print("Clique decomposition creation time: ", time.time() - start_time)
start_time = time.time()
hyper.fit(5, MGG)
print("Execution time: ", time.time() - start_time)
print("Clique decomposition rmse:", hyper.rmse)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=hyper.DELvect, y=hyper.D - hyper.DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error after MDS clique Decomposition")
plt.show()

del hyper


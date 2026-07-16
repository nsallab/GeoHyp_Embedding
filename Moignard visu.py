import cmath

import networkx as nx
from h5py.h5p import PropDCID
from networkx.classes import neighbors
from networkx.generators.geometric import navigable_small_world_graph
from sklearn.metrics.pairwise import pairwise_distances
import HyperboloidMDS as hm
import TreeProcessing as tp
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
import pickle as pkl
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import matplotlib.colors as Colors
import matplotlib.cm as cmx
from matplotlib.collections import PatchCollection
import os

def diffMap(G):
    epsilon=0.3
    nodes2index=dict()
    index2nodes=dict()
    i=0
    weights=np.zeros(G.number_of_nodes())
    for node in G.nodes():
        nodes2index[node]=i
        index2nodes[i]=node
        weights[i] =G.nodes[node]["weight"]
        i +=1
    Mat=nx.adjacency_matrix(G,weight='distance').todense()
    kern=np.exp(-Mat*Mat/epsilon)
    for i in range(kern.shape[0]):
        for j in range(kern.shape[1]):
            if kern[i,j]==1:
                kern[i,j]=0
    kern=kern@np.diag(weights)
    D=np.diag(1/np.sum(kern,axis=1))

    La=D@kern@D
    Da=np.diag(1/np.sum(La,axis=1))
    M=Da@La

    pi=np.sum(kern,axis=1)/np.sum(kern)
    Mtilde=M-np.outer(pi,pi)
    n=M.shape[0]
    MM=np.linalg.inv(np.eye(n)-Mtilde)-np.eye(n)
    pairWise=pairwise_distances(np.diag(1/np.sqrt(pi))@MM)
    aggregateProb={"PS":0.0,"NP":0.0,"HF":0.0,"4SG":0.0, "4SFG":0.0}
    for i in range(pi.shape[0]):
        aggregateProb[G.nodes[index2nodes[i]]["label"]] +=pi[i]

    print(eigenvalues)

def random_walk(G, source, num_steps=10000):
    M=dict()
    current_node=source
    nodes=[]
    transProb=dict()
    i=0
    for node in G.nodes():
        p=np.zeros(G.degree(node))
        i=0
        nodes=[]
        for u in G[node]:
            p[i]=np.exp(-G[node][u]["distance"])
            i += 1
            nodes.append(u)
        p=p/np.sum(p)
        transProb[node]=dict()
        transProb[node]["p"]=p
        transProb[node]["nodes"]=nodes
    count=0
    for node in G.nodes():
        for i in range(10):
            current_node=node
            newcurrent_node = node
            path=[current_node]
            while G.nodes[current_node]["label"] !="4SG" and G.nodes[current_node]["label"] !="4SFG" :
                j=0
                while newcurrent_node in path:
                    if len(set(transProb[current_node]["nodes"]) & set(path)) == len(transProb[current_node]["nodes"]):
                        # we are stuck in a deadend
                        j +=1
                        current_node=path[-(1+j)]
                    newcurrent_node=np.random.choice(transProb[current_node]["nodes"],p=transProb[current_node]["p"])
                path.append(newcurrent_node)
                current_node=newcurrent_node
            ppath=path.copy()
            for p in path:
                if p not in M:
                    M[p]=dict()
                ppath.remove(p)
                for pp in ppath:
                    if pp not in M[p]:
                        M[p][pp]=0
                    M[p][pp] +=1
                    if pp==2197:
                        count +=1
                pass
            pass
        pass

    print(nodes)

def pseudotime(G,root):
  #  l=nx.shortest_path_length(G, source=root, weight='distance')
    categoryDict=dict()
    categoryDict["PS"]=[]
    categoryDict["NP"]=[]
    categoryDict["HF"]=[]
    categoryDict["4SG"]=[]
    categoryDict["4SFG"]=[]
    for node in G.nodes():
        categoryDict[G.nodes[node]["label"]].append(l[node])
    for key in categoryDict:
        categoryDict[key]=np.array(categoryDict[key])
        print(key, "mean",np.mean(categoryDict[key]), "std",np.std(categoryDict[key]))
    pass
from HyperboloidMDS import HyperHyp
def newMergeVertex(u,v,G,addEdges):
    if G.nodes[u]["active"] and G.nodes[v]["active"]:
        if G.nodes[u]["weight"]> G.nodes[v]["weight"]:
            uu=u
            vv=v
        else:
            uu=v
            vv=u
    else:
        if (not G.nodes[u]["active"]) and G.nodes[v]["active"]:
            uu = v
            vv = u
        else:
            if G.nodes[u]["active"] and (not G.nodes[v]["active"]):
                uu = u
                vv = v
            else:
                G[u][v]["active"] = False
                return
    #we merge into uu and remove vv
    G.nodes[vv]["active"]=False
    G.nodes[uu]["name"] +="|"+G.nodes[vv]["name"]
    G.nodes[uu]["weight"]+=G.nodes[vv]["weight"]
    G.nodes[uu]["mix"] +=G.nodes[vv]["mix"]
    for tgt in G[vv]:
        if tgt != uu:
            if tgt in G[uu]:
                #we have a triangle
                G[uu][tgt]["distance"]=min(G[uu][tgt]["distance"],G[vv][tgt]["distance"])
            else:
                G.add_edge(uu,tgt,**G[vv][tgt])
    G.remove_node(vv)


def mergeVertex(u, v, G, addEdges):
    # vv is removed and uu remain
    neighbors=set()
    for tgt in G[vv]:
        if G.nodes[tgt]["active"] and tgt!=uu:
            neighbors.add(tgt)
    for n in neighbors:
        if G.has_edge(n,uu): #edge n to uu exists
            if G.has_edge(n,vv): #edge n to vv exists
                if G[n][uu]["active"]:
                    if G[n][uu]["distance"]<G[n][vv]["distance"]:
                        G[n][vv]["distance"] = G[n][uu]["distance"]
            else: #edge n to vv does not exists
                if n != vv:
                    addEdges.append((n,vv,G[n][uu]))
            G[n][uu]["active"]=False
    return G, addEdges

def approxGraph(G,eps):
    G=G.copy()
    merge=True
    for u in G.nodes():
        G.nodes[u]["mix"]=np.zeros(5)
        if G.nodes[u]['label'] == "PS":
            G.nodes[u]['mix'][0]=1
        if G.nodes[u]['label'] == "NP":
            G.nodes[u]['mix'][1]=1
        if G.nodes[u]['label'] == "HF":
            G.nodes[u]['mix'][2]=1
        if G.nodes[u]['label'] == "4SFG":
            G.nodes[u]['mix'][3]=1
        if G.nodes[u]['label'] == "4SG":
            G.nodes[u]['mix'][4]=1
        G.nodes[u]["active"] = True
    while(merge):
        merge=False
        GG = nx.Graph()
        addEdges=[]
        for u in G.nodes():
            G.nodes[u]["active"] = True
        for u, v, data in G.edges(data=True):
            data["active"] = True
        edgesToRemove=[]
        for u, v, data in G.edges(data=True):
            if data["distance"]<eps:
                edgesToRemove.append((u, v))
        for (u,v) in edgesToRemove:
            if G.has_edge(u,v) and G[u][v]['distance'] < eps:
                # u and v should merge
                merge=True
                newMergeVertex(u, v, G,addEdges)
        for u in G.nodes():
            if G.nodes[u]["active"]:
                GG.add_node(u)
                GG._node[u]=G._node[u].copy()
        for u, v, data in G.edges(data=True):
            if G.nodes[u]["active"] and G.nodes[v]["active"]:
                GG.add_edge(u, v,distance=data['distance'],curv=data['curv'])
        for (u,v,data) in addEdges:
            if GG.has_edge(u,v):
                if GG[u][v]["distance"]<data["distance"]:
                    GG[u][v]['distance'] = data['distance']
                    GG[u][v]['curv'] = data['curv']
            else:
                if G.nodes[u]["active"] and G.nodes[v]["active"]:
                    GG.add_edge(u,v,distance=data['distance'],curv=data['curv'])
        if len(list(nx.connected_components(GG)))>2:
            pass
        G=GG.copy()
    weights=np.zeros(G.number_of_nodes())
    weightsdict=dict()
    ind = 0
    for u in G.nodes():
        G.nodes[u]["mix"]=str(G.nodes[u]["mix"].tolist())
        weights[ind] = G.nodes[u]["weight"]
        weightsdict[u] = G.nodes[u]["weight"]
        ind += 1
    return G,weights, weightsdict

find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e][0] for e in elem]


def read_data(fin, with_labels=True, normalize=False, n_pca=20):
    """
    Reads a dataset in CSV format from the ones in datasets/
    """
    df = pd.read_csv(fin + '.csv', sep=',')
    n = len(df.columns)

    if with_labels:
        x = np.double(df.values[:, 0:n - 1])
        labels = df.values[:, (n - 1)]
        labels = labels.astype(str)
        colnames = df.columns[0:n - 1]
    else:
        x = np.double(df.values)
        labels = ['unknown'] * np.size(x, 0)
        colnames = df.columns

    n = len(colnames)

    idx = np.where(np.std(x, axis=0) != 0)[0]
    x = x[:, idx]

    if normalize:
        s = np.std(x, axis=0)
        s[s == 0] = 1
        x = (x - np.mean(x, axis=0)) / s

    if n_pca:
        if n_pca == 1:
            n_pca = n

        nc = min(n_pca, n)
        pca = PCA(n_components=nc)
        x = pca.fit_transform(x)
        colnames = [f'PC{i+1}' for i in range(n_pca)]

    labels = np.array([str(s) for s in labels])

    return x, labels, colnames

#df=pd.read_csv('/Users/kave/Downloads/PoincareMaps-main-2/datasets/Moignard2015.csv')
data, true_labels, col_names = read_data('/Users/kave/Downloads/PoincareMaps-main-2/datasets/Moignard2015', normalize=False, n_pca=0)
biodf= pd.DataFrame(data=data,columns=col_names,dtype='float64')
biodf['labels'] = true_labels

eryGenes=['HbbbH1', 'Gata1', 'Nfe2', 'Gfi1b', 'Ikaros', 'Myb']
endoGenes=['Erg', 'Sox7', 'Sox17', 'HoxB4', 'Cdh5']
idx_ery = find(col_names, ['HbbbH1', 'Gata1', 'Nfe2', 'Gfi1b', 'Ikaros', 'Myb'])
idx_endo = find(col_names, ['Erg', 'Sox7', 'Sox17', 'HoxB4', 'Cdh5'])


GG=nx.read_graphml("/Users/kave/CLionProjects/NewCurvature/processed/processed.197.graphml")
#tesselation=pkl.load(open("tesselation.pkl","rb"))
weights=[]
for n in GG.nodes():
    if "|" in GG.nodes[n]["name"]:
        GG.nodes[n]["weight"]=len(GG.nodes[n]["name"].split("|"))
    else:
        GG.nodes[n]["weight"]=1
    weights.append( GG.nodes[n]["weight"])
#Aligning bioData with graph nodes after merge
##GG=nx.read_graphml("/Users/kave/Downloads/IranNeighbor1.graphml")
GG=nx.convert_node_labels_to_integers(GG)
G, weights, weightsdict=approxGraph(GG,0.1)
#pseudotime(G,1812 )
#diffMap(G)
datacolumns=list(biodf)+['x','y','z']
datadf=pd.DataFrame(columns=datacolumns,dtype='float64')
ind=0
for u in G.nodes():
    if "|" in G.nodes[u]["name"]:
        indices=[]
        datadf.loc[G.nodes[u]["name"].split("|")[0]] = biodf.iloc[int(G.nodes[u]["name"].split("|")[0])]
    else:
        datadf.loc[G.nodes[u]["name"]]= biodf.iloc[int(G.nodes[u]["name"])]
    ind+=1
datadf["weight"]=weights
#G=GG.copy()
nx.write_graphml(G,"MoignardRicciApprox.graphml")
#nx.write_graphml(G,"IranNeighborApprox.graphml")
index=0
category=np.zeros((len(G.nodes(),)))
weights=np.zeros((len(G.nodes(),)))


globalC=0.8
dim=2
utils=hm.Utils()
MetricGraph = hm.MetricGraph(G)
if os.path.exists("MetricMoignard.pkl"):
    mmMetricGraph = pkl.load(open("MetricMoignard.pkl", "rb"))
else:
    mmMetricGraph = hm.MetricGraph(G.copy())
    s=set(mmMetricGraph.G.nodes())
    anchored=set()
    hyper = hm.HyperHyp(s, mmMetricGraph, anchored, globalC, dim, 1)
    hyper.HYP_fit()
    pkl.dump(mmMetricGraph, open("MetricMoignard.pkl", "wb"))
DELvect = np.zeros(mmMetricGraph.G.number_of_edges())
D = np.zeros(mmMetricGraph.G.number_of_edges())
cnt = 0
unmatched =0
edge=0
vertice=0
for u, v, data in mmMetricGraph.G.edges(data=True):
    D[cnt] =data['edist']
    DELvect[cnt] = data['distance']
    cnt += 1
sserror = np.sum((DELvect - D) ** 2)
rmse = np.sqrt(sserror / G.number_of_edges())

print("Rmse All:", rmse)
print(sserror, rmse)

utils=hm.Utils()
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=DELvect, y=D - DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error after MDS Hyperbolic")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=DELvect, y=D)
plt.xlabel("Target Distance")
plt.ylabel("Estimated Distance")
plt.show()


coords=np.zeros((mmMetricGraph.G.number_of_nodes(),dim+1))
ind = 0
labels=[]
asns=[]
ind2Name={}
for n in mmMetricGraph.G.nodes():
    coords[ind,:]=mmMetricGraph.G.nodes[n]["Coords"]
    if "|" in mmMetricGraph.G.nodes[n]["name"]:
        name=mmMetricGraph.G.nodes[n]["name"].split("|")[0]
    else:
        name=mmMetricGraph.G.nodes[n]["name"]
    ind2Name[ind]=name
    datadf.at[name, 'x']=coords[ind,0]
    datadf.at[name, 'y']=coords[ind,1]
    datadf.at[name, 'z']=coords[ind,2]
    ind+=1

center=datadf['weight'].idxmax()
for n in mmMetricGraph.G.nodes():
    if "|" in mmMetricGraph.G.nodes[n]["name"]:
        name=mmMetricGraph.G.nodes[n]["name"].split("|")[0]
    else:
        name=mmMetricGraph.G.nodes[n]["name"]
    if name==center:
        nodeNum=n
        break
p = nx.shortest_path_length(mmMetricGraph.G, source=nodeNum, weight="distance")
for  n in p:
    if "|" in mmMetricGraph.G.nodes[n]["name"]:
        name=mmMetricGraph.G.nodes[n]["name"].split("|")[0]
    else:
        name=mmMetricGraph.G.nodes[n]["name"]
    datadf.at[name,"category"]=p[n]//0.2

ppCoords=utils.toPP(coords)
from Tesselation import BinaryTiling

tilling = BinaryTiling([np.min(ppCoords[:,0]-0.1) ,np.max(ppCoords[:,0]+0.1)], [np.min(ppCoords[:,1])-0.01,np.max(ppCoords[:,1]+0.01)], 100)
ddatadf=datadf.copy()
ddatadf=ddatadf.reset_index()
for idx in eryGenes:
    ddatadf[idx] = (ddatadf[idx] - ddatadf[idx].min()) / (ddatadf[idx].max() - ddatadf[idx].min())
for idx in endoGenes:
    ddatadf[idx] = (ddatadf[idx] - ddatadf[idx].min()) / (ddatadf[idx].max() - ddatadf[idx].min())

stats=pd.DataFrame(0.0,index=np.arange(len(tilling.cells)) , columns=['PS','NP','HF','4SG','4SFG']+eryGenes+endoGenes)
for index, row in ddatadf.iterrows():
    cell=tilling.returnCell(ppCoords[index,:])
    stats.loc[cell.index,row["labels"]]+=row["weight"]
    for idx in eryGenes:
        stats.loc[cell.index,idx]+=row[idx]*row["weight"]
    for idx in endoGenes:
        stats.loc[cell.index,idx] += row[idx] * row["weight"]
tilling.stats=stats
cmap = plt.get_cmap('inferno')

#import matplotlib.pyplot as plt
fig, ax = plt.subplots()
cmap = cmx.get_cmap('inferno')

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['PS'].sum()
cNorm  = plt.Normalize(vmin=0, vmax=0.5)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.title('PS')
PDcells=[]
colors=[]
for index, row in tilling.stats.iterrows():
#    colorVal = scalarMap.to_rgba(row['PS']/total)
    poly = tilling.cells[index].facePD
#    poly.set_color()
    PDcells.append(poly)
    colors.append(row['PS']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno, norm=cNorm) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('PS')
fig.colorbar(p)
fig.show()

fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['NP'].sum()
#cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['NP'].max()/total)
cNorm  = plt.Normalize(vmin=0, vmax=0.5)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.title('NP')
colors=[]
for index, row in tilling.stats.iterrows():
    colors.append(row['NP']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno,norm=cNorm) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('NP')
fig.colorbar(p)
fig.show()

fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['HF'].sum()
#cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['HF'].max()/total)
cNorm  = plt.Normalize(vmin=0, vmax=0.5)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.title('HF')
colors=[]
for index, row in tilling.stats.iterrows():
    colors.append(row['HF']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno,norm=cNorm) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('HF')
fig.colorbar(p)
fig.show()

fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['4SG'].sum()
#cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['4SG'].max()/total)
cNorm  = plt.Normalize(vmin=0, vmax=1)
plt.title('4SG')
colors=[]
for index, row in tilling.stats.iterrows():
    colors.append(row['4SG']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno,norm=cNorm) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('4SG')
fig.colorbar(p)
fig.show()


fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['4SFG'].sum()
cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['4SFG'].max()/total)
plt.title('4SFG')
colors=[]
for index, row in tilling.stats.iterrows():
    colors.append(row['4SFG']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno, norm=Colors.LogNorm()) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('4SFG')
fig.colorbar(p)
fig.show()

for idx in eryGenes+endoGenes:
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    total = tilling.stats[idx].sum()
    # cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['Cdh5'].max()/total)
    cNorm = plt.Normalize(vmin=0, vmax=0.5)
    plt.title(idx)
    colors = []
    for index, row in tilling.stats.iterrows():
        colors.append(row[idx] / total)
    p = PatchCollection(PDcells, cmap=cmx.inferno, norm=cNorm)  # ,  alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    ax.set_title(idx)
    fig.colorbar(p)
    fig.show()

fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
total=tilling.stats['Gfi1b'].sum()
cNorm  = plt.Normalize(vmin=0, vmax=tilling.stats['Gfi1b'].max()/total)
plt.title('Gfi1b')
colors=[]
for index, row in tilling.stats.iterrows():
    colors.append(row['Gfi1b']/total)
p=PatchCollection(PDcells, cmap=cmx.inferno) #,  alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_title('Gfi1b')
fig.colorbar(p)
fig.show()



N=len(tilling.cells)
rects=[]
colors=dict()
colors['PS']=np.zeros(N)
colors['HF']=np.zeros(N)
colors['NP']=np.zeros(N)
colors['4SG']=np.zeros(N)
colors['4SFG']=np.zeros(N)
colors['Cdh5']=np.ones(N)*(-178.8534)
colors['Gfi1b']=np.ones(N)*(-340.1653)

i=0
for cell in tilling.cells:
    # add rectangle to plot
    rects.append(Rectangle((cell.xmin, cell.ymin), cell.xmax-cell.xmin,cell.ymax-cell.ymin))
    if 'PS' in cell.stats:
        colors['4SG'][i]=cell.stats['4SG']
        colors['4SFG'][i]=cell.stats['4SFG']
        colors['PS'][i]=cell.stats['PS']
        colors['HF'][i]=cell.stats['HF']
        colors['NP'][i]=cell.stats['NP']
        colors['Cdh5'][i] = cell.stats['Cdh5']
        colors['Gfi1b'][i] = cell.stats['Gfi1b']
    i +=1
fig = plt.figure()

#colors = 100*np.random.random(N)
# define Matplotlib figure and axis
ax = fig.add_subplot(421)
ax.set_xlim(tilling.Xmin,tilling.Xmax)
ax.set_ylim(tilling.Ymin,tilling.Ymax)
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(np.log(colors['PS']+1))
ax.add_collection(p)
ax.set_title('PS')
fig.colorbar(p)

ax = fig.add_subplot(422)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim(tilling.Xmin,tilling.Xmax)
ax.set_ylim(tilling.Ymin,tilling.Ymax)
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(np.log(colors['NP']+1))
ax.add_collection(p)
ax.set_title('NP')
fig.colorbar(p)


ax = fig.add_subplot(423)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim([tilling.Xmin,tilling.Xmax])
ax.set_ylim([tilling.Ymin,tilling.Ymax])
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(np.log(colors['HF']+1))
ax.add_collection(p)
ax.set_title('HF')
fig.colorbar(p)

ax = fig.add_subplot(424)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim([tilling.Xmin,tilling.Xmax])
ax.set_ylim([tilling.Ymin,tilling.Ymax])
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(np.log(colors['4SG']+1))
ax.add_collection(p)
ax.set_title('4SG')
fig.colorbar(p)

ax = fig.add_subplot(425)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim([tilling.Xmin,tilling.Xmax])
ax.set_ylim([tilling.Ymin,tilling.Ymax])
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(np.log(colors['4SFG']+1))
ax.add_collection(p)
ax.set_title('4SFG')
fig.colorbar(p)

ax = fig.add_subplot(426)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim([tilling.Xmin,tilling.Xmax])
ax.set_ylim([tilling.Ymin,tilling.Ymax])
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(colors['Cdh5'])
ax.add_collection(p)
ax.set_title('Cdh5')
fig.colorbar(p)

ax = fig.add_subplot(427)
# create simple line plot
#ax.plot([tilling.Xmin,tilling.Xmax], [tilling.Ymin,tilling.Ymax])
ax.set_xlim([tilling.Xmin,tilling.Xmax])
ax.set_ylim([tilling.Ymin,tilling.Ymax])
p = PatchCollection(rects, cmap=matplotlib.cm.inferno, alpha=0.4)
p.set_array(colors['Gfi1b'])
ax.add_collection(p)
ax.set_title('Gfi1b')
fig.colorbar(p)


fig.show()

fig = px.scatter_3d(
    datadf,
    x="x", y="y", z="z",
    color="labels",
    hover_name="labels",
    # color points by label
#    size=np.log(datadf["weight"]+1),
#    size_max=1,
    opacity=0.8
)
fig.update_traces(marker_size = 2)
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis'
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    legend_title="Cell categories"
)

fig.show()
datadf['Cdh5']=datadf['Cdh5'].astype('float64')
datadf['weight']=datadf['weight'].astype('float64')
fig = px.scatter_3d(
    datadf,
    x="x", y="y", z="z",
    color="Cdh5", color_continuous_scale="viridis",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    # color points by label
    size_max=50,
    opacity=0.8
)

fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    legend_title="Cdh5 expression"
)

fig.show()
datadf['Gfi1b']=datadf['Gfi1b'].astype('float64')
fig = px.scatter_3d(
    datadf,
    x="x", y="y", z="z",
    color='Gfi1b', color_continuous_scale="viridis",
    hover_name="labels",
    # color points by label
    size=20*np.log(datadf["weight"]+1),
    size_max=50,
    opacity=0.8
)
fig.show()

fig = px.scatter_3d(
    datadf,
    x="x", y="y", z="z",
    color="category",
    hover_name="labels",
    # color points by label
#    size=np.log(datadf["weight"]+1),
#    size_max=1,
    opacity=0.8
)
fig.update_traces(marker_size = 2)
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    legend_title="Cell distance"
)

fig.show()


ppCoords=utils.toPP(coords)
ind=0
for n in mmMetricGraph.G.nodes():
    if "|" in mmMetricGraph.G.nodes[n]["name"]:
        name=mmMetricGraph.G.nodes[n]["name"].split("|")[0]
    else:
        name=mmMetricGraph.G.nodes[n]["name"]
    datadf.at[name, 'x']=ppCoords[ind,0]
    datadf.at[name, 'y']=ppCoords[ind,1]
    ind +=1

fig = px.scatter(
    datadf, x="x", y="y", color="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    hover_name="labels",
    title="Scatter Plot Colored by category",
)

fig.show()


datadf['Cdh5']=datadf['Cdh5'].astype('float64')

fig = px.scatter(
    datadf, x="x", y="y", color="Cdh5", color_continuous_scale="viridis",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by Cdh5",
)
fig.show()

datadf['Gfi1b']=datadf['Gfi1b'].astype('float64')

fig = px.scatter(
    datadf, x="x", y="y", color="Gfi1b", color_continuous_scale="viridis",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by Gf1b",
)
fig.show()

fig = px.scatter(
    datadf, x="x", y="y", color="category",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by distance",
)
fig.show()


pdCoords=utils.toPD(coords)
ind=0
for n in mmMetricGraph.G.nodes():
    if "|" in mmMetricGraph.G.nodes[n]["name"]:
        name=mmMetricGraph.G.nodes[n]["name"].split("|")[0]
    else:
        name=mmMetricGraph.G.nodes[n]["name"]
    datadf.at[name, 'x']=np.real(pdCoords[ind])
    datadf.at[name, 'y']=np.imag(pdCoords[ind])
    ind +=1

fig = px.scatter(
    datadf, x="x", y="y", color="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    hover_name="labels",
    title="Scatter Plot Colored by category",
)


fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=-1, y0=-1, x1=1, y1=1,
)

fig.show()


datadf['Cdh5']=datadf['Cdh5'].astype('float64')

fig = px.scatter(
    datadf, x="x", y="y", color="Cdh5", color_continuous_scale="viridis",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by Cdh5",
)
fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=-1, y0=-1, x1=1, y1=1,
)

fig.show()

datadf['Gfi1b']=datadf['Gfi1b'].astype('float64')

fig = px.scatter(
    datadf, x="x", y="y", color="Gfi1b", color_continuous_scale="viridis",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by Cdh5",
)
fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=-1, y0=-1, x1=1, y1=1,
)
fig.show()

fig = px.scatter(
    datadf, x="x", y="y", color="category",
    hover_name="labels",
    size=20*np.log(datadf["weight"]+1),
    size_max=30,
    title="Scatter Plot Colored by distance",
)
fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=-1, y0=-1, x1=1, y1=1,
)
fig.show()


cliqueDecomposition = hm.CliqueDecompositionEmbedding(G, 1)
import os.path
if os.path.isfile("treeMoignardApprox.pkl"):
    treeList=[]
    treeList=pkl.load(open("treeMoignardApprox.pkl", "rb"))
#    for u in streeList[:-1]:
#        leftset=u[0]
#        dataset=u[1]
#        s0=set()
#        for e in u[0]:
#            s0.add(int(e[1:]))
#        s1=set()
#        for e in u[1]:
#            s1.add(int(e[1:]))
#        treeList.append([s0,s1,u[2]])
#    s0 = set()
#    for e in streeList[-1]:
#        s0.add(int(e[1:lis]))
#    treeList.append(s0)
else:
    if os.path.isfile("alphaMoignardApprox.pkl"):
        alpha1=pkl.load(open("alphaMoignardApprox.pkl","rb"))
        cliqueDecomposition.alpha=alpha1
    else:
        cliqueDecomposition.complete_to_chordal1()
        pkl.dump(cliqueDecomposition.alpha, open("alphaMoignardApprox.pkl", "wb"))
    cliqueDecomposition.buildCliqueTree()
    tree = cliqueDecomposition.mycliqueTree
    treeList=cliqueDecomposition.tree2List()
    pkl.dump(treeList, open("treeMoignardApprox.pkl", "wb"))
cliqueDecomposition.list2Tree(treeList)

atom=cliqueDecomposition.mycliqueTree
while True:
    if isinstance(atom.right, hm.Tree):
        atom=atom.right
    else:
        break
last=atom
matched=set()
source=next(iter(last.data))
origin=np.zeros(dim+1)
origin[-1]=1
cliqueDecomposition.G.nodes[source]['Coords']=origin
matched.add(source)
MetricGraph = hm.MetricGraph(G)
mMetricGraph = hm.MetricGraph(G.copy())
returnTree=None
tree=last
while tree :
    verticesSet = set(tree.data)
    for s in nx.connected_components(G.subgraph(verticesSet)):
#        comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#        comb.greedyEmbedding3()
        hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
        hyper.HYP_fit()
        matched.update(s)
        print("Atom size center:", len(s))
        print("RMSE Error:", hyper.rmse)
    if  isinstance(tree.right, set):
        verticesSet = set(tree.right).union(set(tree.data))
        for s in nx.connected_components(G.subgraph(verticesSet)):
#            comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#            comb.greedyEmbedding3()
            hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
            hyper.HYP_fit()
            matched.update(s)
            print("Atom size right:", len(s))
            print("RMSE Error:", hyper.rmse)
    verticesSet=set(tree.left).union(set(tree.data))
    for s in nx.connected_components(G.subgraph(verticesSet)):
#        comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#        comb.greedyEmbedding3()
        hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
        hyper.HYP_fit()
        matched.update(s)
        print("Atom size left:", len(s))
        print("RMSE Error:", hyper.rmse)
    tree=tree.father

decompcoords=np.zeros((mmMetricGraph.G.number_of_nodes(),dim+1))
ind = 0
labels=[]
asns=[]
ind2Name={}
for n in mmMetricGraph.G.nodes():
    decompcoords[ind,:]=mmMetricGraph.G.nodes[n]["Coords"]
    datadf.loc[name]['x']=decompcoords[ind,0]
    datadf.loc[name]['y']=decompcoords[ind,1]
    datadf.loc[name]['z']=decompcoords[ind,2]
    ind+=1

fig = px.scatter_3d(
    datadf,
    x="x", y="y", z="z",
    color="labels",
    hover_name="labels",
    # color points by label
#    size=np.log(datadf["weight"]+1),
#    size_max=1,
    opacity=0.8
)
fig.update_traces(marker_size = 1)
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    legend_title="Cell categories"
)

fig.show()

pddecompCoords=utils.toPD(decompcoords)
datadf['x']=np.real(pddecompCoords)
datadf['y']=np.imag(pddecompCoords)
datadf['Gfi1b']=datadf['Gfi1b'].astype('float64')

fig = px.scatter(
    datadf, x="x", y="y", color="Gfi1b", color_continuous_scale="viridis",
    hover_name="labels",
    size=np.log(datadf["weight"]+1),
    size_max=20,
    title="Scatter Plot Colored by Cdh5",
)
fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=-1, y0=-1, x1=1, y1=1,
)

fig.show()


newCoords=utils.hypMoveCenter(coords.transpose(),coords[8,:]).transpose()
df['x']=newCoords[:,0]
df['y']=newCoords[:,1]
df['z']=np.log(newCoords[:,2])
df['label']=labels
df['asn']=asns


fig = px.scatter_3d(
    df,
    x="x", y="y", z="z",
    color="label",
    hover_name="asn",
    # color points by label
#    size_max=0.01,
    opacity=0.8
)
fig.update_traces(marker_size = 3)
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    legend_title="Countries"
)

fig.show()


ppNewCoords=utils.toPP(newCoords*1e-5)

df=pd.DataFrame()
df['x']=ppNewCoords[:,0]
df['y']=ppNewCoords[:,1]
df['label']=labels
df['asn']=asns
fig = px.scatter(
    df, x="x", y="y", color="label",
    hover_name="asn",
    title="Scatter Plot Colored by Label",
)

fig.show()



Z=utils.toPD(coords)
newZ=utils.toPD(newCoords)
df=pd.DataFrame()
df['x']=np.real(Z)
df['y']=np.imag(Z)
df['label']=labels
df['asn']=asns
fig = px.scatter(
    df, x="x", y="y", color="label",
    hover_name="asn",
    title="Scatter Plot Colored by Label",
)
ZZ=utils.toPS(coords*1e5)

fig.show()

df['x']=np.real(newZ)
df['y']=np.imag(newZ)
fig = px.scatter(
    df, x="x", y="y", color="label",
    hover_name="asn",
    title="Scatter Plot Colored by Label",
)
fig.show()



#
eng = matlab.engine.start_matlab("-desktop")
# eng.cd(r'/Users/kave/Downloads/MDS-PD-matlab', nargout=0)
eng.workspace['X1'] = eng.double(coords)
# eng.workspace['ymax'] = eng.double(coords[:,1].ravel())
# eng.workspace['zmax'] = eng.double(coords[:,-1].ravel())
eng.workspace['category'] = eng.double(category.ravel())
eng.workspace['weight'] = eng.double(weights.ravel())

# #eng.workspace['xpmax'] = eng.double(hyperMDS_max.X[:,0].ravel())
# #eng.workspace['ypmax'] = eng.double(hyperMDS_max.X[:,1].ravel())
# #eng.workspace['zpmax'] = eng.double(hyperMDS_max.X[:,2].ravel())
eng.workspace['Zxpdmax'] = eng.double(np.real(Z).ravel())
eng.workspace['Zypdmax'] = eng.double(np.imag(Z).ravel())
# #eng.workspace['sourcesMax']=eng.double(np.array(sourcesMax))
# #eng.workspace['targetsMax']=eng.double(np.array(targetsMax))



cliqueDecomposition = hm.CliqueDecompositionEmbedding(G, 1)
import os.path
if os.path.isfile("treeMoignardApprox.pkl"):
    treeList=[]
    treeList=pkl.load(open("treeMoignardApprox.pkl", "rb"))
#    for u in streeList[:-1]:
#        leftset=u[0]
#        dataset=u[1]
#        s0=set()
#        for e in u[0]:
#            s0.add(int(e[1:]))
#        s1=set()
#        for e in u[1]:
#            s1.add(int(e[1:]))
#        treeList.append([s0,s1,u[2]])
#    s0 = set()
#    for e in streeList[-1]:
#        s0.add(int(e[1:lis]))
#    treeList.append(s0)
else:
    if os.path.isfile("alphaMoignardApprox.pkl"):
        alpha1=pkl.load(open("alphaMoignardApprox.pkl","rb"))
        cliqueDecomposition.alpha=alpha1
    else:
        cliqueDecomposition.complete_to_chordal1()
        pkl.dump(cliqueDecomposition.alpha, open("alphaMoignardApprox.pkl", "wb"))
    cliqueDecomposition.buildCliqueTree()
    tree = cliqueDecomposition.mycliqueTree
    treeList=cliqueDecomposition.tree2List()
    pkl.dump(treeList, open("treeMoignardApprox.pkl", "wb"))
cliqueDecomposition.list2Tree(treeList)

atom=cliqueDecomposition.mycliqueTree
while True:
    if isinstance(atom.right, hm.Tree):
        atom=atom.right
    else:
        break
last=atom
matched=set()
source=next(iter(last.data))
origin=np.zeros(dim+1)
origin[-1]=1
cliqueDecomposition.G.nodes[source]['Coords']=origin
matched.add(source)
MetricGraph = hm.MetricGraph(G)
mMetricGraph = hm.MetricGraph(G.copy())
returnTree=None
tree=last
while tree :
    verticesSet = set(tree.data)
    for s in nx.connected_components(G.subgraph(verticesSet)):
#        comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#        comb.greedyEmbedding3()
        hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
        hyper.HYP_fit()
        matched.update(s)
        print("Atom size center:", len(s))
        print("RMSE Error:", hyper.rmse)
    if  isinstance(tree.right, set):
        verticesSet = set(tree.right).union(set(tree.data))
        for s in nx.connected_components(G.subgraph(verticesSet)):
#            comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#            comb.greedyEmbedding3()
            hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
            hyper.HYP_fit()
            matched.update(s)
            print("Atom size right:", len(s))
            print("RMSE Error:", hyper.rmse)
    verticesSet=set(tree.left).union(set(tree.data))
    for s in nx.connected_components(G.subgraph(verticesSet)):
#        comb = hm.CombinatorialEmbedding(s, MetricGraph, globalC, 2, 1)
#        comb.greedyEmbedding3()
        hyper = hm.HyperHyp(s, mMetricGraph, matched, globalC, dim, 1)
        hyper.HYP_fit()
        matched.update(s)
        print("Atom size left:", len(s))
        print("RMSE Error:", hyper.rmse)
    tree=tree.father

DELvect = np.zeros(G.number_of_edges())
D = np.zeros(G.number_of_edges())
cnt = 0
unmatched =0
edge=0
vertice=0
coords=np.zeros((G.number_of_nodes(),dim+1))
cnt =0
for u in MetricGraph.G.nodes():
    coords[cnt,:]=mMetricGraph.G.nodes[u]['Coords']
    cnt +=1
cnt=0
for u, v, data in mMetricGraph.G.edges(data=True):
    D[cnt] =data['edist']
    DELvect[cnt] = data['distance']
#    if data['type']=='u':
#        unmatched +=1
#    if data['type']=='e':
#        edge += 1
#    if data['type']=='v':
#        vertice += 1
    cnt += 1
sserror = np.sum((DELvect - D) ** 2)
rmse = np.sqrt(sserror / G.number_of_edges())
print("Rmse Clique Decomposition:", rmse)
#print("unmatched:", unmatched, "edgeMatch:", edge,"verticeMatch:", vertice)
print("Curvature:",globalC)
Z=utils.toPD(coords)

#eng = matlab.engine.start_matlab("-desktop")
eng.cd(r'/Users/kave/Downloads/MDS-PD-matlab', nargout=0)
eng.workspace['X2'] = eng.double(coords)
#eng.workspace['ymax'] = eng.double(coords[:,1].ravel())
#eng.workspace['zmax'] = eng.double(coords[:,-1].ravel())
eng.workspace['category'] = eng.double(category.ravel())
#eng.workspace['weights'] = eng.double(weights.ravel())

#eng.workspace['xpmax'] = eng.double(hyperMDS_max.X[:,0].ravel())
#eng.workspace['ypmax'] = eng.double(hyperMDS_max.X[:,1].ravel())
#eng.workspace['zpmax'] = eng.double(hyperMDS_max.X[:,2].ravel())
eng.workspace['Zxpdmax'] = eng.double(np.real(Z).ravel())
eng.workspace['Zypdmax'] = eng.double(np.imag(Z).ravel())
#eng.workspace['sourcesMax']=eng.double(np.array(sourcesMax))
#eng.workspace['targetsMax']=eng.double(np.array(targetsMax))





#print(sserror, rmse)
s=set(mMetricGraph.G.nodes())
anchored=set()
hyper = hm.HyperHyp(s, mMetricGraph, anchored, globalC, dim, 1)
hyper.HYP_fit(coords[:,0:dim])
DELvect = np.zeros(G.number_of_edges())
D = np.zeros(G.number_of_edges())
cnt = 0
unmatched =0
edge=0
vertice=0
for u, v, data in mMetricGraph.G.edges(data=True):
    D[cnt] =data['edist']
    DELvect[cnt] = data['distance']
    cnt += 1
sserror = np.sum((DELvect - D) ** 2)
rmse = np.sqrt(sserror / G.number_of_edges())

print("Rmse after recalibration :", rmse)
#print(sserror, rmse)


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=DELvect, y=D - DELvect)
plt.xlabel("Target Distance")
plt.ylabel("error after MDS Hyperbolic")
plt.show()
coords=np.zeros((mMetricGraph.G.number_of_nodes(),dim+1))
labels=np.zeros(mMetricGraph.G.number_of_nodes())
ind = 0
values=["PS","NP","HF","4SFG","4SG"]
for n in G.nodes():
    coords[ind,:]=mMetricGraph.G.nodes[n]["Coords"]
    ind+=1
Z=utils.toPD(coords)

#eng = matlab.engine.start_matlab("-desktop")
#eng.cd(r'/Users/kave/Downloads/MDS-PD-matlab', nargout=0)
eng.workspace['X3'] = eng.double(coords)
#eng.workspace['ymax'] = eng.double(coords[:,1].ravel())
#eng.workspace['zmax'] = eng.double(coords[:,-1].ravel())
#eng.workspace['category'] = eng.double(category.ravel())
#eng.workspace['xpmax'] = eng.double(hyperMDS_max.X[:,0].ravel())
#eng.workspace['ypmax'] = eng.double(hyperMDS_max.X[:,1].ravel())
#eng.workspace['zpmax'] = eng.double(hyperMDS_max.X[:,2].ravel())
eng.workspace['Zxpdmax'] = eng.double(np.real(Z).ravel())
eng.workspace['Zypdmax'] = eng.double(np.imag(Z).ravel())
#eng.workspace['sourcesMax']=eng.double(np.array(sourcesMax))
#eng.workspace['targetsMax']=eng.double(np.array(targetsMax))




#maxTree
print("Max Tree Process")
treeMax=tp.maximumSpanningTree(G)
sourcesMax=[]
targetsMax=[]
for u,v in treeMax.edges():
    sourcesMax.append(u)
    targetsMax.append(v)


Tlist,levels,roots=tp.setDepth([treeMax])
intreeMax=tp.distFilt(Tlist[0],0.0)
MG=hm.MetricGraph(G)
verticeSet=list(MG.G.nodes())

root=roots[0]







#astretch,avstretchsq, aerrsq, gdist, mintdist, minvstretch,minerr=getDistorsion(G,tree)
#print("MinTree:",astretch,avstretchsq,aerrsq)
Tlist,levels,roots=tp.setDepth([treeMax])
intreeMax=tp.distFilt(Tlist[0],0.0)
MG=hm.MetricGraph(G)
verticeSet=list(MG.G.nodes())

root=roots[0]
hyper=hm.HyperHyper()
#root=3070
#treeGraph_max,coords=hyperMDS_max.sarkar_tree(Tlist[0], 1891, 1)
#treeGraph_max,root,coords,=hyper.greedyEmbedding2(MG.G, 0.2)
treeGraph_max,coords,=hyper.greedyEmbedding2(MG.G, 0.2)
utils=hm.Utils()
(sse,rmse, srse,rrmse,D, estD)=hm.Utils.distorsion(MG.G,0.2)
Z=utils.toPD(coords)
print("Greedy Gradient rmse:", rmse)
print("Greedy Gradient rmsre:", rrmse)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=D, y=estD-D)
plt.xlabel("Target Distance")
plt.ylabel("error after embedding Hyperbolic")
plt.show()

eng = matlab.engine.start_matlab("-desktop")
eng.cd(r'/Users/kave/Downloads/MDS-PD-matlab', nargout=0)
eng.workspace['X'] = eng.double(coords)
#eng.workspace['ymax'] = eng.double(coords[:,1].ravel())
#eng.workspace['zmax'] = eng.double(coords[:,2].ravel())
eng.workspace['category'] = eng.double(category)
#eng.workspace['xpmax'] = eng.double(hyperMDS_max.X[:,0].ravel())
#eng.workspace['ypmax'] = eng.double(hyperMDS_max.X[:,1].ravel())
#eng.workspace['zpmax'] = eng.double(hyperMDS_max.X[:,2].ravel())
eng.workspace['Zxpdmax'] = eng.double(np.real(Z).ravel())
eng.workspace['Zypdmax'] = eng.double(np.imag(Z).ravel())
eng.workspace['sourcesMax']=eng.double(np.array(sourcesMax))
eng.workspace['targetsMax']=eng.double(np.array(targetsMax))
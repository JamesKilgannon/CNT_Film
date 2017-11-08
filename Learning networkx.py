
# coding: utf-8

# In[50]:

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import networkx as nx

get_ipython().magic('matplotlib inline')


# In[ ]:




# In[17]:

G = nx.Graph()
G.add_nodes_from("ABCDEF")


# In[21]:

print(G.number_of_edges())
print(G.number_of_nodes())


# In[22]:

# my old graph format:
# An undirected graph. Each dictionary entry lists the nodes that have a connection to the key.
graph2 = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'F'],
    'D': ['B', 'E', 'F'],
    'E': ['D', 'F'],
    'F': ['C', 'D', 'E'],
}
# The resistances in graph2
graph2_r = {
    'AB': 10,
    'AC': 1,
    'BC': 3,
    'BD': 100,
    'CF': 1,
    'DE': 4,
    'DF': 1,
    'EF': 15,
}


# In[46]:

R = nx.Graph()
R.add_nodes_from("ABCDEF")
for node in graph2:
    for conn in graph2[node]:
        str = ''.join(sorted([node, conn]))
        try:
            resistance = graph2_r[str]
        except KeyError:
            resistance = None
            raise
        R.add_edge(node, conn, weight=resistance)


# In[49]:

R.edges['D', 'B']


# In[45]:




# In[53]:

nx.draw(R, with_labels=True, font_weight='bold')


# In[ ]:





# coding: utf-8

# In[1]:

import numpy as np


# Let's get some code down! 
# 
# I want to model the resistance through a network of resistors, where the resistance is only in the connections between the individual resistors.
# 
# But I can perhaps start with the connections being perfectly conducting and the resistance being a material property of the individual resistors. That seems like it would be easier.
# 
# What I need to know:
# * Fundamentals of graph theory (?)
# * Resistance in a network of resistors - this seems like something with well-established math around it
#     * yikes this is complicated
# 

# ---
# Let's say we want to represent the situation in a film of CNT as a mathematical graph (with nodes and lines). Make some simplifying assumptions:
# 1. There is a resistance at each **node** (this corresponds with our greatly simplified picture of the physical reality)
# 2. There is no resistance in the lines; just in the connections
# 3. Each CNT intersects with exactly two others
# 
# Developing this idea:
# * There are only ever two CNT involved at each node, meaning maximum four lines connected to the node.
#     * Problem: resistance should be zero when staying on the same CNT but not zero when jumping from one to another.
#     
# 
# Oh goodness this is very complex.
# 
# Ok figure that out later, let's code *something*. 
# 
# 
# [Wikipedia: Nodal Analysis](https://en.wikipedia.org/wiki/Nodal_analysis)
# 
# Nodal analysis is based on Kirchhoff's Current Law, which says the sum of currents at a node is zero. Note positive currents come toward the node and negative currents go away from the node.
# $$V=IR$$
# $$I = \frac{V}{R}$$
# 

# Sample network
# 
# ![img](https://dl2.pushbulletusercontent.com/vbQLyBgJMAF46sICGIOBqQGxrFnbvJFe/IMG_8700.JPG)

# In[2]:

# implementing some form of nodal analysis


#First we need a graph.
#example directed graph from https://www.python.org/doc/essays/graphs/
graph1 = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
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

#vvvvvvv [[[[you can't do this]]]] vvvvvvv
#graph2_r = {
#     ['A', 'B']: 10.,
#     ['A', 'C']: 1.,
#     ['B', 'C']: 3.,
#     ['B', 'D']: 100.,
#     ['C', 'F']: 1.,
#     ['D', 'E']: 4.,
#     ['D', 'F']: 1.,
#     ['E', 'F']: 15.
#}

def get_connections(graph, node):
    """
    Returns a list of two-element lists representing the connections
    of the specified node
    """
    # given the node, find the list of all nodes to which it is connected    
    connected = graph[node]
    # convert each connection to the two-letter string in graph_r    
    connections = []
    for n in connected:
        connections.append([node] + [n])
    
    return connections
    
def get_rs(lines, r_dict):
    """
    Given a list of nodal connections (as a list of two-element lists),
    returns a corresponding list of resistances
    """
    rs = []
    for line in lines:
        
        str = ''.join(sorted(line))
        try:
            rs.append(r_dict[str])
        except KeyError:
            #print("Error: no resistance value found for connection {}".format(line))
            rs.append(None)
            raise
    return rs
    
def nodal_analysis(graph, node):
    """
    graph should be a dictionary with nodes as keys
    node should be one of the nodes in graph
    """
    
    
    #hang on, I'm pausing mid-programming here. see below.
    
    


# In[ ]:




# A couple of hours on a Sunday night is not enough time to learn circuit theory. I need to retain my sanity.
# 
# For "Nodal Analysis" we need to know where current is being applied to the whole system. We can do this by fixing the current at the two nodes between which the current is applied.
# 
# Let's imagine we apply a 2A current from B to D, making 
# $$
# i_B = -2A
# $$
# and 
# $$
# i_D = +2A
# $$

# In[3]:

# dict containing voltages at each node, default 0
graph2_v = {
        'A': 0.,
        'B': 0.,
        'C': 0.,
        'D': 0.,
        'E': 0.,
        'F': 0.}


# We want to make a matrix in the following form:
# 
#  | | | | 
# ---|---|---|---|---
# G_11|G_12|G_13|...|G_1N
# G_21|G_22|...|...|G_2N
# ...|...|...|...|...
# G_N1|G_N2|...|...|G_NN
# 
# Where in this case we'll use the DC simplification
# $G=\frac{1}{R}$
# 
# And we know $G_{ii}$ is the sum of $G$s connected to node $i$
# 
# ~~And also that $G_{ij}$ is the negative sum of $G$s between $i$ and $j$~~
# 
# From the paper ["Modeling percolation..."](http://www.mdpi.com/1996-1944/8/10/5334), we read that $G_{ij}=0$ except when $i$ and $j$ have a *direct* connection by one resistor. When they do have a direct connection, $G_{ij} = -\frac{1}{R_{ij}}$
# 
# (note, this may be a simplification the authors of that paper used[{?}])

# In[4]:



def G_matrix(graph, graph_r):
    """
    Returns a matrix of the conductances G[i][j], given a graph and 
    a listing of the resistances of edges in the graph.
    """
    num_nodes = len(graph)
    G = np.zeros((num_nodes, num_nodes))
    ii=0
    for i in graph:
        jj=0
        for j in graph:


            if i==j: # i.e. if this element is on the diagonal
                Gij = 1./sum(get_rs(get_connections(graph, i),graph_r))
                G[ii][jj] = Gij
            elif ii<jj: # i.e. if we're above the diagonal
                #print([i,j])
                # this sets Gij to the negative reciprocal of Rij if there is a 
                # direct connection between nodes i and j. If not, it leaves it 0.
                try:
                    resistance = get_rs([[i,j]], graph_r)
                    #print(resistance)
                    Gij = 1./resistance[0]
                    G[ii][jj] = Gij
                except KeyError:
                    pass # (it's already set to zero)
            else: # i.e. if we're below the diagonal
                if not G[jj][ii] == 0.:
                    G[ii][jj] = G[jj][ii] # because the matrix is symmetric
            jj += 1
        ii += 1    
    return G



# To calculate the equivalent resistance between two points in the graph, here's a cno

# In[5]:

###############################################################
###############################################################
###############################################################
# code below here is not useful with the re-thought G matrix! #
###############################################################
###############################################################
###############################################################


# In[6]:

# for reference 
graph2 = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'F'],
    'D': ['B', 'E', 'F'],
    'E': ['D', 'F'],
    'F': ['C', 'D', 'E'],
}


# In[7]:

def paths_len_2(graph):
    """ 
    Returns a list of lists of strings, containing all the possible
    2-length paths in the graph. Does not repeat any paths,
    but does allow paths that are the reverse of each other.
    """
    paths = []
    nodes_completed = []
    for node in graph:
        for n in graph[node]:
            #if not n in nodes_completed:
                paths.append([node, n])
        nodes_completed.append(node)
    return paths


# In[ ]:




# In[8]:

[['A', 'B'],
 ['A', 'C'],
 ['B', 'C'],
 ['B', 'D'],
 ['C', 'F'],
 ['D', 'E'],
 ['D', 'F'],
 ['E', 'F']]


# In[9]:

def all_paths(graph):
    """
    Finds all of the paths in the graph. Excludes paths which visit any node more
    than once. Does not exclude paths which are the reverse of other paths.
    """
    all_paths = []
    # uses the paths_len_2 function to get started
    paths_n = paths_len_2(graph)

    all_paths += paths_n

    # this loop will do all the >2 length paths
    for i in range(3, len(graph)+1):
        paths_len_n = []
        for path in paths_n:
            t_connections = graph[path[-1]]
            for n in t_connections:
                if not n in path:
                    paths_len_n.append(path + [n])
        all_paths += paths_len_n
        paths_n = paths_len_n

    return all_paths


# In[10]:

#%%timeit
#all_paths(graph2)


# In[11]:

###############################################################
###############################################################
###############################################################
# code above here is not useful with the re-thought G matrix! #
###############################################################
###############################################################
###############################################################


# In[12]:

graph2_r_alt = {
    'AB': 1.,
    'AC': 1.,
    'BC': 1.,
    'BD': 1.,
    'CF': 1.,
    'DE': 1.,
    'DF': 1.,
    'EF': 1.,
}


# In[13]:


# Set our known conditions
graph2_v['B'] = -2.
graph2_v['D'] = 2.

G = G_matrix(graph2, graph2_r_alt)

graph = graph2
test_nodes = ['E', 'B']

#Put a 1A current between the nodes we're concerned with
I = np.zeros(len(graph))
first_I = True
k = 0
assert len(test_nodes) == 2
for node in graph:
    if node in test_nodes:
        if first_I:
            I[k] = 1.
            first_I = False
        else:
            I[k] = -1.
    k += 1
V = np.linalg.solve(G, I)

print("Currents: {}".format(I))
print("Node voltages: {}".format(V)) # is the voltage at each node.

# Now we use the voltages at the two nodes of interest
# to get the equivalent resistance. We don't have to divide by 
# the current since we used a 1A current.
equivalent_resistance = abs(sum(I*V)) # numpy matrix operations make this really simple!
print("The calculated equivalent resistance is {} V".format(equivalent_resistance))


# In[ ]:




# In[ ]:




# In[ ]:




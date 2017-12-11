
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import pylab #displays arrays as images for easy error checking
import numpy as np
import scipy as sp
import networkx as nx
from PIL import Image
from PIL import ImageDraw

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp


# In[2]:

# Functions for equivalent resistance calculation
def G_matrix(graph):
    """
    Using the data from the input networkx graph, returns
    the G matrix needed to complete the nodal analysis calculation.
    """
    # Find the reciprocal of resistance for each connection (i.e. each edge)
    for node1, node2 in graph.edges():
        graph[node1][node2]['reciprocalR'] = 1.0 / graph[node1][node2]['resistance']
    # The adjacency matrix gives all needed elements but the diagonal
    G = nx.adjacency_matrix(graph, weight='reciprocalR')
    # Add the diagonal
    G.setdiag(np.squeeze(np.asarray(-nx.incidence_matrix(graph, weight='reciprocalR').sum(axis=1))))
    # G is a csr_matrix, but we want an array
    return G.toarray()

def equivalent_resistance(graph, check_nodes):
    """
    Given a graph and a list of two check nodes,
    computes the equivalent resistance.
    """
    # Get the G matrix
    G = G_matrix(graph)
    I = np.zeros(len(G))
    I[check_nodes[0]] = 1.
    I[check_nodes[1]] = -1.
    
    # Solve for the voltage matrix
    try:
        V = np.linalg.solve(G, I)
        
        # use a simple numpy operation to compute the equivalent resistance
        equivalent_resistance = abs(sum(I*V))
        return equivalent_resistance
    except:
        # if np.linalg.solve fails, raise an error
        print("Error: could not solve the matrix equation. Is the G-matrix singular?")
        raise

# credit to Bryce Boe http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(ax, ay, bx, by, cx, cy):
    """
    Determines whether points a, b, and c are counerclockwise
    """
    return (cy - ay)*(bx - ax) > (by-ay)*(cx-ax)

def intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """
    a and b describe one line segment; c and d describe another.
    """
    return (ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and
            ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy))

def model(network_size,
         CNT_length_normal,
         CNT_length_stddev,
         CNT_num_tubes,
         resistance_mean,
         resistance_stddev):
    """
    Returns the equivalent resistance, given inputs of the size of the CNT film,
    and information about the tubes' size distribution.
    """
    #converting CNT_num_tubes to an integer so function doesn't break when the array
    #for sensitivity analysis is passed to it
    CNT_num_tubes = int(CNT_num_tubes)
    CNT_init = np.zeros((CNT_num_tubes+2,7))

    #creating the pseudo tubes that will act as the edges in the network
    CNT_init[0:2,:] = [[network_size,0,0,0,0,network_size,0],
                       [network_size,0,1,0,network_size,network_size,network_size]]
    
    #Generating tube information
    #randomly assigning tube lengths distributed around a set tube length
    logmean = np.log(CNT_length_normal / (np.sqrt(1+(CNT_length_stddev/CNT_length_normal)**2)))
    logstdev = np.sqrt(np.log(1+(CNT_length_stddev/CNT_length_normal)**2))
    CNT_init[2:,0] = np.random.lognormal(logmean, logstdev, CNT_num_tubes)

    #randomly assign starting point and orientation
    CNT_init[2:,1:4] = np.random.rand(CNT_num_tubes, 3)

    #applying scaling to random numbers so they match the needed values
    scaling_factor = np.array([1, network_size, network_size, 2*np.pi, 1, 1, 1])
    CNT_init = CNT_init * scaling_factor

    #calculating the x-max for the tubes
    CNT_init[:,5] = CNT_init[:,1] + np.cos(CNT_init[:,3]) * CNT_init[:,0]

    #calculating the y-max for the tubes
    CNT_init[:,6] = CNT_init[:,2] + np.sin(CNT_init[:,3]) * CNT_init[:,0]

    #calculating slope
    CNT_init[:,3] = np.tan(CNT_init[:,3])
    
    #calculating the y-intercept of the lines
    CNT_init[:,4] = CNT_init[:,2] - CNT_init[:,3] * CNT_init[:,2]

    #print CNT_init to file, indexing to the run number (hopefully)
    filename = "CNT_data_" + str(sampleindex) + ".npy"
    np.save(filename, CNT_init)
    
    #generating a boolean array of the tubes that intersect
    CNT_intersect = np.zeros((CNT_num_tubes+2,CNT_num_tubes+2),dtype=bool)
    
    for i,row1 in enumerate(CNT_init):
        for j,row2 in enumerate(CNT_init[i+1:,:]):
            coords = np.concatenate((row1[1:3], row1[5:7], row2[1:3], row2[5:7]))
            if intersect(*coords):
                CNT_intersect[i,j+i+1] = True
    
    #gives the indicies along the x-axis of the true values as the 
    #first array and the y-values as the second array
    CNT_tube_num1, CNT_tube_num2 = np.where(CNT_intersect)
    
    #add the intersections as edges in a networkx graph
    graph = nx.Graph()
    graph.add_edges_from((CNT_tube_num1[k], CNT_tube_num2[k],
                          {'resistance': np.random.normal(resistance_mean, resistance_stddev)})
                         for k in range(0, np.sum(CNT_intersect)))
    
    #get rid of any bits of the graph not contiguous with node 0 (one of the test nodes)
    #thanks to Pieter Swart from 2006 [https://groups.google.com/forum/#!topic/networkx-discuss/XmP5wZhrDMI]
    try:
        contiguous_nodes = nx.node_connected_component(graph, 0)
        new_graph = graph.subgraph(contiguous_nodes)
    except KeyError:
        print("Could not compute equivalent resistance; the starting tube has no intersections")
        path_exists = False
    
    #computes equiv. resistance
    try:
        path_exists = nx.has_path(new_graph, 0, 1)
    except:
        path_exists = False
    
    if path_exists:
        try:
            eqr = equivalent_resistance(new_graph,[0,1])
        except:
            eqr = np.nan
    else:
        print("Could not compute equivalent resistance; there is no contiguous path through the network.")
        eqr = np.nan
 
    return eqr


## read the six parameters from the input file
#
## run the model:
#equivalent_resistance = model(network_size, CNT_length_normal, CNT_length_stddev,
#                                CNT_num_tubes, resistance_mean, resistance_stddev)
#
## save the model output
#
## (unfinished)
#
#
##Important variables
#network_size = 10 #side length of network boundaries
#CNT_length_normal = 445 #normal length of CNT at center of distribution
#CNT_length_stddev = 310 #standard deviation of CNT length from normal
#CNT_num_tubes = 1000 #number of tubes in film
#resistance_mean = 10
#resistance_stddev = 1


# In[3]:

#Defining the problem
morris_problem = {
    # There are six variables
    'num_vars': 6,
    # These are their names
    'names': ['network_size', 'CNT_length_normal', 'CNT_length_stddev', 'CNT_num_tubes', 'resistance_mean', 'resistance_stddev'],
    # These are their plausible ranges over which we'll move the variables
    'bounds': [[1000, 10000], # network size (nm)
               [100, 1000], # length that tubes are distributed around (nm)
               [100, 500], # standard deviation of tube length (nm)
               [100, 5000], # number of tubes in network
               [1, 100], # mean contact resistance of tubes (Ω)
               [1, 20], # standard deviation of contact resistance (Ω)
              ],
    # I don't want to group any of these variables together
    'groups': None
    }


# In[4]:

num_levels = 50
grid_jump = 2
sample = ms.sample(morris_problem, 1000, num_levels, grid_jump)
#forcing the number of tubes to be an integer so no errors are thrown
#sample[:,3] = np.around(sample[:,3])


# In[5]:

print(sample.shape)
print(sample)


# In[6]:

##run this to recreate error
#sample = np.load('sample_input_error.npy')
##shows the variables that will return the error
#print(sample.shape)
#print(sample)


# In[7]:

print(sample.shape[0])
print(sample[0])
for row in sample:
    print(row)

# Run the sample through the monte carlo procedure of the power model
#for sampleindex in range(0,sample.shape[0]):
#    output = model(*sample.T[:,sampleindex])
#    print(sampleindex)
#    print(sample.T[:,sampleindex])
#    print(output)

equivalent_resistance_results = []
for sampleindex, parameters in enumerate(sample):
    equivalent_resistance_results.append(model(*parameters))
    
    #remove the line below this one
    print("Finished run index {}".format(sampleindex))
    
assert len(equivalent_resistance_results) == sample.shape[0]

np.save("CNT_results.npy", np.array(equivalent_resistance_results))

#output = model(*sample.T[:,1])
#print(output.shape)

#print(max(output))
# In[ ]:




# In[8]:

np.save("CNT_parameters.npy", sample)


# In[9]:

print((np.load("CNT_parameters.npy").shape))


# # Analysis section

# In[39]:

# import all of the CNT data files 
filepath = "/Users/mplajer/Dropbox/Documents/Northeastern/CHME 5137 - Computational Modeling/Project - CNT Films/CNT_Film/CNT/CNT_results_"


for index in range(350):
    load_filename = filepath + str(index) + ".npy"
    array = (np.load(load_filename))
    if index == 0:
        results = array
    else:
        results = np.concatenate((results, array))
print(results.shape)


# In[40]:

np.save("CNT_results_combined.npy", results)


# In[ ]:




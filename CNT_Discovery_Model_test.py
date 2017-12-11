import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import pylab #displays arrays as images for easy error checking
import os
import numpy as np
import scipy as sp
import networkx as nx
from PIL import Image
from PIL import ImageDraw

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

# steps to actually run the thing!:

#0: initialize
job_number = int(os.getenv('SLURM_ARRAY_TASK_ID', default='0'))
number_of_runs_per_job = 3


#1: get the parameters file into a np array
sample = np.load("CNT_parameters.npy")

#2: run the model for each line of the np array
equivalent_resistance_results = []
for sampleindex in range(job_number * number_of_runs_per_job, (job_number * number_of_runs_per_job)+3):
    parameters = sample[sampleindex]
    equivalent_resistance_results.append(model(*parameters))

assert len(equivalent_resistance_results) == number_of_runs_per_job

#3: save the results
job_filename = "CNT_results_" + str(job_number) + ".npy"
np.save(job_filename, np.array(equivalent_resistance_results))

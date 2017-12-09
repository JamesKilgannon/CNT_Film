
# coding: utf-8

# In[1]:

from matplotlib import pyplot as plt
from matplotlib import pylab #displays arrays as images for easy error checking
import numpy as np
import scipy as sp
import networkx as nx
from PIL import Image
from PIL import ImageDraw

get_ipython().magic('matplotlib inline')


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
    G.setdiag(np.squeeze(np.asarray(-nx.incidence_matrix(graph, 
                                                          weight='reciprocalR').sum(axis=1))))
    # G is a csr_matrix, but we want an array
    return G.toarray()

def equivalent_resistance(graph, check_nodes):
    """
    Given a graph and a list of two check nodes,
    computes the equivalent resistance.
    """

    # Get the G matrix
    G = G_matrix(graph)
    print("G matrix is got!")
    # Get the matrix of currents, I
    #I = np.zeros(CNT_num_tubes) - problem with this: we don't want to include tubes with no intersections
    #I = np.zeros(graph.number_of_edges()) - alsp problem but I don't know why
    I = np.zeros(len(G))
    print("Length of G matrix: {}".format(len(G)))
    I[check_nodes[0]] = 1.
    I[check_nodes[1]] = -1.
  
    #####################################
    ## Remove after debugging finishes ##
    #####################################
    print(G)                      #######
    print(I)                      #######
    #####################################
    ## Remove after debugging finishes ##
    #####################################
    
    # Solve for the voltage matrix
    try:
        V = np.linalg.solve(G, I)
        
        #####################################
        ## Remove after debugging finishes ##
        #####################################
        print(V)                           
        #####################################
        ## Remove after debugging finishes ##
        #####################################
        
        # use a simple numpy operation to compute the equivalent resistance
        equivalent_resistance = abs(sum(I*V))
        return equivalent_resistance
    except:
        # if np.linalg.solve fails, raise an error
        print("Error: could not solve the matrix equation. Is the G-matrix singular?")
        raise


# In[1]:

#Important variables
network_size = 10 #side length of network boundaries
CNT_length_normal = 50 #normal length of CNT at center of distribution
CNT_length_stddev = 5 #standard deviation of CNT length from normal
CNT_num_tubes = 1000 #number of tubes in film

def model(network_size,
         CNT_length_normal,
         CNT_length_stddev,
         CNT_num_tubes):
    """
    Returns the equivalent resistance, given inputs of the size of the CNT film,
    and information about the tubes' size distribution.

    """
    CNT_init = np.zeros((CNT_num_tubes+2,7))

    #creating the pseudo tubes that will act as the edges in the network
    CNT_init[0:2,:] = [[network_size,0,0,0,0,network_size,0],
                       [network_size,0,1,0,network_size,network_size,network_size]]
    
    #Generating tube information
    #randomly assigning tube lengths distributed around a set tube length
    CNT_init[2:,0] = np.random.normal(CNT_length_normal, CNT_length_stddev, CNT_num_tubes)
    
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
    
    #generating a boolean array of the tubes that intersect
    CNT_intersect = np.zeros((CNT_num_tubes+2,CNT_num_tubes+2),dtype=bool)
    
    i=0
    for row1 in CNT_init:
        m1 = row1[3]
        b1 = row1[4]
        j=i+1
        for row2 in CNT_init[i+1:,:]:
            #check for parallel
            m2 = row2[3]
            b2 = row2[4]
            if m1 == m2:
                CNT_intersect[i,j] = False
                continue
            x_intersect = (b2 - b1) / (m1 - m2)
            if row1[1] <= x_intersect <= row1[5] and row2[1] <= x_intersect <= row2[5]:
                CNT_intersect[i,j] = True
            j+=1
        i+=1
    
    ########
    ########
    #(debugging)         
   # pylab.imshow(CNT_intersect)
    ########
    
    #gives the indicies along the x-axis of the true values as the 
    #first array and the y-values as the second array
    CNT_tube_num1, CNT_tube_num2 = np.where(CNT_intersect)
    
    #add the intersections as edges in a networkx graph
    graph = nx.Graph()
    graph.add_edges_from((CNT_tube_num1[k], CNT_tube_num2[k],
                          {'resistance': 10.}) for k in range(0, np.sum(CNT_intersect)))
    
    #get rid of any bits of the graph not contiguous with node 0 (one of the test nodes)
    #thanks to Pieter Swart from 2006 [https://groups.google.com/forum/#!topic/networkx-discuss/XmP5wZhrDMI]
    contiguous_nodes = nx.node_connected_component(graph, 0)
    new_graph = graph.subgraph(contiguous_nodes)
    
    ########
    ########
    #(debugging)
    nx.draw(new_graph, with_labels=True, font_weight='bold', node_size=100, font_size=9)
    print("Node 0 has {} neighbors".format(len(list(new_graph.neighbors(0)))))
    print("---")
    print("Node 1 has {} neighbors".format(len(list(new_graph.neighbors(1)))))
    print("---")
    print("Continuous path present: {}".format(nx.has_path(new_graph, 0, 1)))
    print("Shape of G matrix: {}".format(np.shape(G_matrix(new_graph))))
    ########
    
    #computes equiv. resistance
    if nx.has_path(new_graph, 0, 1):
        try:
            eqr = equivalent_resistance(new_graph,[0,1])
        except:
            eqr = np.nan
    else:
        print("Could not compute equivalent resistance; there is no contiguous path through the network.")
        eqr = np.nan
 
    return eqr


# In[4]:

# this gives the situation where the equiv. resistance comes to 10.0
np.random.seed(57)
print(model(10, 10, 1, 5000))


# In[5]:


numsuccesses = 0
numtens = 0
for i in range(30):
    print("--------------------")
    try:
        eqr = model(10, 10, 1, 1000)
        print(eqr)
        if not eqr == np.nan:
            numsuccesses += 1
        if eqr == 10:
            numtens += 1
    except:
        pass
print("numsuccesses = {}".format(numsuccesses))
print("numtens = {}".format(numtens))


# In[6]:

# Right now the challenge is this: it seems that often, the result is an equivalent resistance of exactly 10,
# which seems to probably be wrong.


# In[ ]:

#This cell will create and save the generated network as "CNT_network.png" when run
#size of image
image_size = (network_size, network_size) #pixles
#initializing a blank image
image = Image.new('RGBA', image_size, (255,255,255,255))
#selecting the image in which to draw and creating the drawing interface
draw = ImageDraw.Draw(image)
#setting the color for each line as black
color  = (0, 0, 0, 255) 

#drawing the individual line segment on the image
for tube in CNT_endpoints:
    draw.line(((tube[0],tube[1]),(tube[2],tube[3])), fill=color, width=1)

#dislplaying the image
plt.imshow(np.asarray(im), origin='lower')
plt.show()
image.save('CNT_network.png')


# In[ ]:




# In[ ]:




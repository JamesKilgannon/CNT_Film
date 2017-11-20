
# coding: utf-8

# In[1]:

import numpy as np


# In[ ]:

#Important variables
network_size = 1000 #side length of network boundaries


# In[ ]:

def CNT():
    #randomly assign length distributed around a set tube length
        #need to average tube length and distribution
    #randomly assign starting point
    #randomly assign angle between 0 and 180
    


# In[2]:

def EOL(endpoint_1,endpoint_2):
    #Determines the slope, y-intercept of the parent line, and range of x-values of a line
    #segment made by 2 points. Endpoints are input as a list of x and y values e.g. [x,y]
    x1 = endpoint_1[0]
    y1 = endpoint_1[1]
    x2 = endpoint_2[0]
    y2 = endpoint_2[1]
    
    #Preventing undefined values for infinite slope
    if x1 == x2:
        slope = 1e10
    else:
        slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    #finding range of x-values the segment runs for to make determining intersection easier
    x_range = [np.min(x1,x2),np.max(x1,x2)]
    return slope, y_intercept, x_range


# In[3]:

def intersect(EOL_1,EOL_2):
    #Calculates if two line segments intersect given two equations of lines given the
    #slopes, y-intercepts, and acceptable range
    #Input format: [slope, y_intercept, x_range]
    m1 = EOL_1[0] #slope of line 1
    m2 = EOL_2[0] #slope of line 2
    b1 = EOL_1[1] #Y-intercept of line 1
    b2 = EOL_2[1] #Y-intercept of line 2
    x_range_1 = EOL_1[2]
    x_range_2 = EOL_2[2]
    
    #Checking for parallel
    if m1 == m2:
        return False
    
    x_intersect = (b2 - b1) / (m1 - m2)
    
    return x_intersect in range(*x_range_1) and x_intersect in range(*x_range_2)


# In[ ]:

def y_delta


# In[29]:

1e6


# In[ ]:




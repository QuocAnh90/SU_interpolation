# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:03:12 2022

@author: Ivan Depina
Norwegian University of Science and Technology
ivan.depina@ntnu.no
"""
# Import modules---------------------------------------------------------------
import numpy as np
import os
from scipy.interpolate import RBFInterpolator, interp1d
import matplotlib.pyplot as plt

# File locations-------------------------------------------------------
location= os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Folder with files
fileLoc=location+"\\Gjerdrum boreholes\\"

# Read boreholes files---------------------------------------------------------
# Open the file with borFile
borFile=open(fileLoc+'Boreholes_moved_origin.txt','r')

# Counter
numBor=-1

# Borehole data
boreholeData=[]

for line in borFile:
    if(numBor==-1):
        # Skip the first line
        # Update the counter
        numBor+=1

    else:
        # Split the string
        boreholeData.append(line.split())
        # Update the counter
        numBor+=1

# Close the file
borFile.close()

# Interpolate borehole data vertically-----------------------------------------

# Define a list with observations
observations=[]

# Interpolation of borehole data vertically
nVer=200

# Read the remaining files to build a database
for i in range(numBor):
    # Open the borehole file
    dataIn=open(fileLoc+boreholeData[i][0]+'.txt','r')
    # Skip the first line
    dataIn.readline()
    
    # Vertical values of su
    su=[]
    
    for data in dataIn:
        # Append su values
        su.append([data.split()[0],data.split()[1]])
    
    # Close the file
    dataIn.close()
    
    # Interpolate su values vertically
    # Allocate array
    su=np.array(su,dtype=np.float64)
    # Build a linear interpolation model
    suInterp=interp1d(su[:,0],su[:,1])
    # Discretize along depth
    zVal=np.linspace(np.min(su[:,0]),np.max(su[:,0]),num=nVer)
    # Interpolate su values
    suVal=suInterp(zVal)
    
    # Append values to the observation list
    for j in range(nVer):
        # Append values to the observation list
        observations.append([boreholeData[i][1],boreholeData[i][2],zVal[j],
                             suVal[j]])

# Convert list to an array
observations=np.array(observations,dtype=np.float64)

# Interpolate borehole data on the rest of the domain--------------------------

# Define the bounding box
# Lower bound
lBound=np.min(observations[:,0:3],0)
# Upper bound
uBound=np.max(observations[:,0:3],0)

# Discretize the spatial domain
nx=50; ny=50; nz=50

# Define 1D discretization
x=np.linspace(lBound[0]*0.9,uBound[0]*1.1,num=nx)
y=np.linspace(lBound[1]*0.9,uBound[1]*1.1,num=ny)
z=np.linspace(lBound[2]*0.5,uBound[2],num=nz)

# Generate mesh of points
xx,yy,zz=np.meshgrid(x,y,z)

# Stack the points vertically
points=np.vstack((np.reshape(xx,(1,nx*ny*nz)),np.reshape(yy,(1,nx*ny*nz)),
                  np.reshape(zz,(1,nx*ny*nz)))).T

# Predict values at points with a linear RBF model
predictionRBF=RBFInterpolator(observations[:,0:3],observations[:,3],
                              kernel='linear',degree=1, epsilon=10)(points)


# Plot values------------------------------------------------------------------
# Plot all values
fig = plt.figure()
ax = plt.axes(projection='3d')
p = ax.scatter(points[:,0], points[:,1], points[:,2], c=predictionRBF[:], 
               cmap='viridis', linewidth=0.5, vmin=20,vmax=220)
fig.colorbar(p)

# Create cross section along x axis
idx=(points[:,0]>180)*(points[:,0]<200)==1

fig = plt.figure()
ax = plt.axes(projection='3d')
p = ax.scatter(points[idx,0], points[idx,1], points[idx,2], 
               c=predictionRBF[idx], cmap='viridis', linewidth=0.5, 
               vmin=20,vmax=220)
p = ax.scatter(observations[:,0], observations[:,1], observations[:,2], 
               c=observations[:,3], cmap='viridis', linewidth=0.5, vmin=20,
               vmax=220)
fig.colorbar(p)

# Create cross section along z axis
idz=(points[:,1]>600)*(points[:,1]<620)==1

# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
p = ax.scatter(points[idz,0], points[idz,1], points[idz,2], 
               c=predictionRBF[idz], cmap='viridis', linewidth=0.5, vmin=20,
               vmax=220)
p = ax.scatter(observations[:,0], observations[:,1], observations[:,2], 
               c=observations[:,3], cmap='viridis', linewidth=0.5, vmin=20,
               vmax=220)
fig.colorbar(p)

# End--------------------------------------------------------------------------

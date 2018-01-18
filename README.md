# sysid2
The repo deals with the Reduced Order System Identification Of Task-Space Controlled Industrial Robots. Batch processing of control output and task velocity data is considered for this identification process.

## Overview.ipynb
### Reading of all log data
The step_log folder containing all the log files must be transferred to the folder where program execution is taking place. The program sorts all the log files in two parts. In the first part, all the '.log' files are taken into account. In the latter part, '.log.\<n>' files are sorted. These two parts are then combined together in a single array consisting of all sorted log files.
The total number of positions and total number of observations in each positions are calculated as:

![equation](http://latex.codecogs.com/gif.latex?Positions%20%3D%20%5Csum%20%7B%27.log%27%7D) 

![equation](http://latex.codecogs.com/gif.latex?Observations%20%3D%20%5Cfrac%7B%5Csum%7B%27.log.%3Cn%3E%27%7D%7D%7BPositions%7D)

The log files are then read into a 2D dataframe array i.e. dataframe[position][observation].

### Plotting of all data
Using subplots, all the observations under a particular position are plotted in a single image file and is saved under a folder named Overview. The number of generated image files is in accordance with the number of available positions.

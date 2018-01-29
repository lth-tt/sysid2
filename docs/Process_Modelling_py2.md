# Reduced Order System Identification of batch data using Python 2 

Program input : 

                Control Output and Task Velocity log data

Program output: 

                Plot of all data saved inside a folder Overview as *.png files
                Plot of all general models for each experimental task space point as *.png file
                Statistical data of general as well as all models as *.txt files and dataframes
                State space parameters of all general models as *.txt file
 
![Program Flow](process_operation.png?raw=true "Title")


## Reading of all log data
The step_log folder containing all the log files must be transferred to the folder where program execution is taking place. The program sorts all the log files in two parts. In the first part, all the '.log' files are taken into account. In the latter part, '.log.\<n>' files are sorted. These two parts are then combined together in a single array consisting of all sorted log files.
The total number of positions and total number of observations in each positions are calculated as:

![equation](http://latex.codecogs.com/gif.latex?Positions%20%3D%20%5Csum%20%7B%27.log%27%7D) 

![equation](http://latex.codecogs.com/gif.latex?Observations%20%3D%20%5Cfrac%7B%5Csum%7B%27.log.%3Cn%3E%27%7D%20&plus;%20Positions%7D%7BPositions%7D)

The log files are then read into a 2D dataframe array i.e. dataframe[position][observation].

## Plotting of all data
Using subplots, all the observations under a particular position are plotted in a single image file and is saved under a folder named Overview. The number of generated image files is in accordance with the number of available positions.


## Savitzky-Golay Smoothing
The smoothing operation is invoked by calling the smoothing_sg() function.
The function returns two arrays having smoothed values according to first order and second order systems.
The output step response data along with the required order(1 or 2) are passed as arguments. 
The data values which are considerably greater than 0, are smoothed.  


## PT1 and PT2 Estimation
The first and second order parametric estimation of the smoothed model is carried out using the pt1() and pt2() functions.
It accepts the smoothed data and calculates the system parameters like steady state value, time constant and delay in a pt1 and pt2 
system and additionally, zeta in case of pt2 system. These functions also returns the transfer function model of each of model responses.
Based on the calculated parameters, the model step response values are passed into an array and is then plotted.


## Ideal Model Calculation
A generalised model is construted on each position and is named as ideal model. This ideal model, based on a particular position,
is obtained by taking the average of all the models from each observations under that particular position. The operation is carried 
out through ideal_pt1() and ideal_pt2 functions. The ideal models from all positions are then plotted.

## Results
Based on the developed models, the statistical information and state space parameter matrices are generated. For easy accessment, the 
statistical data is passed to dataframes df_ideal and df_all. This statistical data along with the state space parameters are then 
outputted into a *.txt file.


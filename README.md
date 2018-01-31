# sysid2
The repo deals with the Reduced Order System Identification Of Task-Space Controlled Industrial Robots. Batch processing of control output and task velocity data is considered for this identification process.

## File  and Folder Information

     docs/:           contains the general documentation
### Batch data
     step_log/:       contains log data of 14 observations in 9 positions
     step_log_new/:   contains log data of 10 observations in 9 positions
### Program files
     View_data_py2.ipynb:              Plots all the step response log data using Python 2
     View_data_py3.ipynb:              Plots all the step response log data using Python 3 
     identification_py2.py:            Python file containing all the functions. (Python version 2)
     identification_py3.py:            Python file containing all the functions. (Python version 3) 
     Process_Modeling_py2.ipynb:       System Identification using python 2. Depends on identification_py2.py.
     Process_Modeling_py3.ipynb:       System Identification using python 3. Depends on identification_py3.py.
### Output files
     model_pt1.png:                    Modeling based on pt1 estimation
     model_pt2.png:                    Modeling based on pt2 estimation
     Statistical_Output.txt:           Statistical data of all the ideal models in 9 positions         
     State_Space_Parameters.txt:       Text file containing all the model state space parameters
     All_Model_Statistical_Output.txt: Statistical data of all the observations in 9 positions


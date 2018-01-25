#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.signal as sig #for running savgol filter
import math #for finding window size
from sklearn.metrics import mean_squared_error
import control as con





#the function reads a single input and output data and returns a dataframe having all the information 
def read_data(control_output, task_vel):
    df_soll     = pd.read_csv(control_output, header = 0, names = ['time', 'x_soll'])
    df_soll     = df_soll.set_index('time')
    df_soll     = df_soll[~df_soll.index.duplicated(keep = 'first')] 
    #gets rid of any duplicate index values if present
    df_ist      = pd.read_csv(task_vel, header = 0, names = ['time', 'x_ist'])
    df_ist      = df_ist.set_index('time')
    df_ist      = df_ist[~df_ist.index.duplicated(keep = 'first')]
    df_ist_soll = pd.concat([df_soll.x_soll, df_ist.x_ist], axis = 1).fillna(method = 'pad')
    df_ist_soll = df_ist_soll.fillna(0)
    return df_ist_soll





#the function reads the step response data from the files and outputs a dataframe
def batch_read_data(control_output, task_vel):

    #the function removes the negative trend which could be a possibility in some step responses
    def trend_remove(df_i_s):#df_i_s: dataframe_ist_soll
        df_i_s['trend'] = np.sign(df_i_s['x_ist'].rolling(window=5).mean().diff().fillna(0)).map({0:'FLAT',1:'UP',-1:'DOWN'})
        #A new column named trend is added to the dataframe having either one of 'UP', 'FLAT' and 'DOWN' values
        rev = list(df_i_s.trend.values)[::-1]
        #The trend values are reversed so as to count the number of 'DOWN' trend at the end of the observation
        counter = 0

        for i in range(0,len(rev)):
            if rev[i] == 'DOWN':
                counter = counter + 1
            else:
                break
        leng = len(df_i_s)
        if counter > 5: #if the trend is noticable, then the dataframe is updated.
            df_i_s = df_i_s.head(leng - counter)
        return df_i_s

    try: #try except is used to ignore empty data
        df_soll     = pd.read_csv(control_output, header = 0)
        df_soll.columns = df_soll.columns.str.strip() #removing whitespace from header name
        df_soll.rename(columns={'values[0]': 'x_soll', 'time':'time_old', 'local_time':'time'}, inplace=True)
        #local time from the log file is the actual time. 'time' from the log data is not used and thereby is renamed to time_old. 
        df_soll     = df_soll.set_index('time')
        df_ist      = pd.read_csv(task_vel, header = 0)
        df_ist.columns = df_ist.columns.str.strip()
        df_ist.rename(columns={'values[2]': 'x_ist', 'time':'time_old', 'local_time':'time'}, inplace=True)
        df_ist      = df_ist.set_index('time')
        df_ist_soll = pd.concat([df_soll.x_soll, df_ist.x_ist], axis = 1).fillna(method = 'pad')
        df_ist_soll = df_ist_soll.fillna(0)
        df_ist_soll = trend_remove(df_ist_soll)
        df_ist_soll.drop('trend', axis = 1, inplace = True) #removing the extra trend column from the dataframe
        return df_ist_soll
    except:
        pass


    
    
#function that strips unwanted zeros and changes the dataframe response to a standard unit step response
def strip_multiply(dataframe):
    
    time_array                     = []
    no_zero_df                     = [] #dataframe having non zero x_soll values
    input_array                    = []
    output_array                   = []
    time_of_step                   = [] #time at which the step occurs
    last_zero_df                   = [] #dataframe before step change (x_soll value is the last zero)
    dataframe_new                  = [] #dataframe having positive x_ist
    data_for_modeling              = [] #consists of modified x_soll, x_ist and time values
    response_start_time            = [] #time at which the new response starts
    x_soll_steady_state            = [] #chooses the last x_soll value(time indexed) as the stead_state value
    multiplication_factor          = [] 
    #calculates the factor to be multiplied with each of the responses for getting a unit step response
    data_with_initial_zero         = [] #returns a step response starting from a single zero
    time_series_starting_from_zero = [] #consists of a new time series data whose time starts from zero
    
    for i in range(0,len(dataframe)):
        dataframe_new.append(dataframe[i][dataframe[i].x_ist > 0])
        no_zero_df.append(dataframe_new[i][dataframe_new[i].x_soll > 0])                      
        time_of_step.append(no_zero_df[i].index[0]) 
        last_zero_df.append(dataframe_new[i][(dataframe_new[i].x_soll == 0) & \
                                             (dataframe_new[i].index < time_of_step[i])].tail(1)) 
        #tail function selects the last x_soll zero value

        response_start_time.append(pd.concat([last_zero_df[i], no_zero_df[i]]).index[0]) 
        data_with_initial_zero.append(pd.concat([last_zero_df[i], no_zero_df[i]])) 
        time_series_starting_from_zero.append(data_with_initial_zero[i].index - response_start_time[i]) 
        data_for_modeling.append(data_with_initial_zero[i].set_index(time_series_starting_from_zero[i]))
        #changing the index of the dataframe with the new time series

        x_soll_steady_state.append(no_zero_df[i].x_soll.head(1)) 
        multiplication_factor.append(1 / (pd.Series(x_soll_steady_state[i]).values[0]))
        input_array.append((multiplication_factor[i] * data_for_modeling[i].x_soll).tolist())                    
        output_array.append((multiplication_factor[i] * data_for_modeling[i].x_ist).tolist()) 
        time_array.append(data_for_modeling[i].index.tolist()) 
    return input_array, output_array, time_array





#the function accepts an order and the time series to output the aic, mse and fitted values of the series
def order_ar_P(ar_order, output):
    ar_system     = ARIMA(output, order=(ar_order, 0, 0))
    fitted_ar     = ar_system.fit().fittedvalues
    fitted_ar[0]  = 0 #assigning the first fitted output value to zero since the initial value always resulted in an abrupt value
    mse_ar        = mean_squared_error(output, fitted_ar)
    output_length = len(output) 
    aic_ar        = (output_length * np.log(mse_ar)) + (2 * ar_order) + (output_length * 1 * (np.log(2 * np.pi) + 1))
    return aic_ar, mse_ar, fitted_ar





#function that smoothes the data using a savgol filter
def smooth(fitted_values, order):
    
    c = 0 #a counter that returns the number of zeros present in the dataframe
    
    for i in range(0,len(fitted_values)):
        if fitted_values[i] < 0.02: 
            #there are some values which are in the range of 0.01 and these values are negligible when compared to the whole data
            c = c + 1
        else:
            break

    fitted_without_zeros  = fitted_values[c:] 
    #smoothing is done after stripping of the zeros
    multiplication_factor = (len(fitted_without_zeros) / 2) % 2 
    #calculates the factor to be multiplied in finding the window length
    #window length must always be an odd number. 
   
    if multiplication_factor   == 0.0:
        window_length     = math.ceil((len(fitted_without_zeros) / 2) + 1)
    elif multiplication_factor == 0.5:
        window_length     = math.ceil((len(fitted_without_zeros) + 1) / 2)
    else:
        window_length     = math.ceil((len(fitted_without_zeros) / 2) + (2 * multiplication_factor))
        #returns an odd integer, which is greater than half of the length of fitted data 
    
    filter_output         = sig.savgol_filter(fitted_without_zeros, window_length, order) 
    #the filter preserves the distributional features like relative maxima, minima and width of the time series
    smoothed_data         = np.append(fitted_values[0:c], filter_output)
    return smoothed_data 





#The function calculates mse between data and model. model_time samples are lesser than data_time. We take account of data_output corresponding to model output. So matching is necessary.   
def mse(data_output, data_time, model_output, model_time):
    data_output_df = pd.concat([pd.DataFrame(data_output, columns = ['data_output']),
                                pd.DataFrame(data_time, columns = ['time'])], axis = 1)
    model_output_df = pd.concat([pd.DataFrame(model_output, columns = ['model_output']),
                                 pd.DataFrame(model_time, columns = ['time'])], axis = 1)
       
    #dt_match_mt = data_time matching with model_time
    dt_match_mt = [] 
    #2d array because similar match values are passed as separate array inside the main array. Main array has a size as that of model_array.
    for i in range(0, len(model_time)):
        dt_match_mt.append([])
        for j in range(0, len(data_time)):
            if round(abs(model_time[i]))    == round(abs(data_time[j])) or \
                (round(model_time[i])       == round(data_time[j] + 0.5)) or \
                (round(model_time[i] + 0.5) == round(data_time[j])): 
                #allows matching of times with 0.5 accuracy
                dt_match_mt[i].append(data_time[j])
    
    #if the difference between model_time elements and matching data_time elements is minimum, then such values are passed into array 
    least_difference = [] 
    for i in range(0, len(model_time)):
        least_difference.append(min(abs(model_time[i] - dt_match_mt[i])))
    
    #data_time corresponding to model_time
    data_time_sliced = []
    for i in range(0, len(model_time)):
        for j in range(0, len(data_time)):
            if abs(model_time[i] - data_time[j]) == least_difference[i]:
                data_time_sliced.append(data_time[j])
    
    #data_output corresponding to data_time
    data_output_sliced = []
    for i in range(0, len(model_time)):
        data_output_sliced.append(list(data_output_df.data_output   \
                                       [data_output_df.time == data_time_sliced[i]])[0])
   
    mse = mean_squared_error(data_output_sliced, model_output)
    return mse





''' Alternate Parameter Estimation Methods for the following pt1 and pt2 functions can be found in 
K. J. Astrom and T. Hagglund. PID Controllers: Theory,  Design, and Tuning. Instrument Society of America,  Triangle  Research Park, N.C., 2nd edition, 1995. (Chapter 2)
&
H. Rake. Step response and frequency response 
methods. Automatica, 16:522-524, 1980. 
'''
#the function estimates the first order parameters and returns it along with the transfer function
def pt1(smooth, time):
    smoothed_df     = pd.concat([pd.DataFrame(smooth, columns = ['smoothed']), \
                                 pd.DataFrame(time, columns   = ['time'])], axis = 1)
    steady_state    = smooth[-1] 
    #last element of the smoothed data is passed as steady state value
    standard_output = steady_state * (1 - np.exp(-1)) 
    #case when t = time_constant in the eqn. 
    #############################################################################
    #       standard_output = steady_state * (1 - e ^ (−t / time_constant))     #
    #############################################################################
    delay         = smoothed_df.time[smoothed_df.smoothed < 0.02].values[-1]
    #returns the time at which the step change occurs i.e a transition from 0 to some value.
    #Values lesser than 0.02 were treatred as zero.
    time_constant = smoothed_df.time[smoothed_df.index == abs(smoothed_df.smoothed - standard_output)\
                                     .sort_values().index[0]].values[0]    
    #derived from the equation of standard_output
    tf_pt1                 = con.matlab.tf(steady_state, [time_constant, 1])
    numerator, denominator = con.pade(delay, 1)
    delay_tf_pt1           = con.matlab.tf(numerator,denominator)
    yout_pt1, t_pt1        = con.matlab.step(tf_pt1 * delay_tf_pt1)
    #first order transfer function is given by
    ###############################################################################
    #                                  steady_state * (e ^ - (delay * s))         #
    #    transfer_function(s) =       ------------------------------------        #
    #                                       (time_constant * s + 1 )              #
    ###############################################################################
    return tf_pt1 * delay_tf_pt1, yout_pt1, t_pt1, delay, time_constant, steady_state





'''
The following pt2 function is based on the research aricle:
C. Huang and C. Chou, "Estimation of the underdamped second-order parameters from the system transient",
Industrial & Engineering Chemistry Research, vol. 33, no. 1, pp. 174-176, 1994. 
However, the delay calculation is based on visual inspection method as stated in http://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/AC2-08-System_Identification.pdf 
'''
#the function estimates the second order parameters and returns it along with the transfer function
def pt2(smooth, time):
    
    def fourpoint(z):
        f1_zeta = 0.451465 + (0.066696 * z) + (0.013639 * z ** 2)
        f3_zeta = 0.800879 + (0.194550 * z) + (0.101784 * z ** 2)
        f6_zeta = 1.202664 + (0.288331 * z) + (0.530572 * z ** 2)
        f9_zeta = 1.941112 - (1.237235 * z) + (3.182373 * z ** 2)
        return f1_zeta, f3_zeta, f6_zeta, f9_zeta
    
    def method(): 
        #the second method from the article is adopted, as the response/data handled strictly follows this method. 
        zeta = np.sqrt((np.log(overshoot) ** 2) / ((np.pi ** 2) + (np.log(overshoot) ** 2)))
        
        f1_zeta, f3_zeta, f6_zeta, f9_zeta = fourpoint(zeta)
        
        time_constant = (t9 - t1) / (f9_zeta - f1_zeta) 
        #delay        = t1 - time_constant*f1_zeta              #based on article.
        delay         = smoothed_df.time[smoothed_df.smoothed < 0.02].values[-1]  
        return zeta, time_constant, delay
    
    smoothed_df  = pd.concat([pd.DataFrame(smooth, columns = ['smoothed']), \
                              pd.DataFrame(time, columns = ['time'])], axis = 1)
    steady_state = smooth[-1]
    
    #ssn = steady state at n/10th instant of time   
    ss1 = steady_state * 0.1
    ss3 = steady_state * 0.3
    ss6 = steady_state * 0.6
    ss9 = steady_state * 0.9
    
    #tn = time at n/10th instant
    t1 = smoothed_df.time[smoothed_df.index == abs(smoothed_df.smoothed - ss1).sort_values().index[0]].values[0]
    t3 = smoothed_df.time[smoothed_df.index == abs(smoothed_df.smoothed - ss3).sort_values().index[0]].values[0]
    t6 = smoothed_df.time[smoothed_df.index == abs(smoothed_df.smoothed - ss6).sort_values().index[0]].values[0]
    t9 = smoothed_df.time[smoothed_df.index == abs(smoothed_df.smoothed - ss9).sort_values().index[0]].values[0]
       
    peak = smoothed_df.smoothed.max() 
    #returns the highest output in the response
    overshoot = (peak - steady_state) / steady_state 
    #represented as Mp in article
        
    zeta, time_constant, delay = method()
   
    tf_pt2 = con.matlab.tf(steady_state, [time_constant ** 2, 2 * zeta * time_constant, 1])
    n_2, d_2 = con.pade(delay, 1)
    delay_tf_pt2 = con.matlab.tf(n_2, d_2)
    yout_pt2,t_pt2 = con.matlab.step(tf_pt2 * delay_tf_pt2)
    #second order transfer function is given by
    ########################################################################################################
    #                                           steady_state * (e ^ - (delay * s))                         #
    #    transfer_function(s) =  ----------------------------------------------------------------------    #
    #                            ((time_constant ^ 2) * (s ^ 2)) + (2 * zeta * time_constant * s) + 1 )    #
    ########################################################################################################
    return tf_pt2 * delay_tf_pt2, yout_pt2, t_pt2, delay, time_constant, steady_state, zeta





#the function calculates the ideal model in a pt1 system based on the steady state(ss), time constant(tc) and delay(d) values
def ideal_pt1(ss_array, tc_array, d_array):
    ideal_ss                    = np.average(ss_array)
    ideal_tc                    = np.average(tc_array)
    ideal_d                     = np.average(d_array)
    ideal_tf_pt1                = con.matlab.tf(ideal_ss, [ideal_tc, 1])
    numerator, denominator      = con.pade(ideal_d, 1)
    ideal_d_tf_pt1              = con.matlab.tf(numerator,denominator)
    ideal_yout_pt1, ideal_t_pt1 = con.matlab.step(ideal_tf_pt1 * ideal_d_tf_pt1)
    return ideal_tf_pt1 * ideal_d_tf_pt1, ideal_yout_pt1, ideal_t_pt1





#the function calculates the ideal model in a pt2 system based on the steady state(ss), time constant(tc), delay(d) and zeta(z) values
def ideal_pt2(ss_array, tc_array, d_array, z_array):
    ideal_ss                    = np.average(ss_array)
    ideal_tc                    = np.average(tc_array)
    ideal_d                     = np.average(d_array)
    ideal_z                     = np.average(z_array)
    ideal_tf_pt2                = con.matlab.tf(ideal_ss, [ideal_tc ** 2, 2 * ideal_z * ideal_tc, 1])
    numerator, denominator      = con.pade(ideal_d, 1)
    ideal_d_tf_pt2              = con.matlab.tf(numerator,denominator)
    ideal_yout_pt2, ideal_t_pt2 = con.matlab.step(ideal_tf_pt2 * ideal_d_tf_pt2)
    return ideal_tf_pt2 * ideal_d_tf_pt2, ideal_yout_pt2, ideal_t_pt2





#the function returns the state space(ss) parameters for a given transfer function(tf)
def ss(tf):
    ss_parameters = con.matlab.tf2ss(tf)
    return ss_parameters





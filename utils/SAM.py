"""
SAM
~~~

Supportive functions for algorithms in TOR
"""
from __future__ import division
from copy import copy
import csv
import math
import sys

import numpy as np
from scipy.integrate import simps
from scipy.stats import linregress
from pandas import DataFrame

from utilikilt.integrate import fast_integrate

# XXX remove this
__version__="1.2.4"


def find_mean_flow_from_pef(flow, pef, t_offset):
    """
    Find the mean flow from our pef to end of expiration
    """
    for idx, vol in enumerate(flow):
        if vol == pef:
            # Advance the index to account for the time offset
            idx = idx + int(t_offset / .02)
            break

    # filter out anything over -3
    # Wait should we do this? This has potential to catch copd.
    #remaining_flow = filter(lambda x: x <= -3, flow[idx:])
    remaining_flow = flow[idx:]
    if len(remaining_flow) == 0:
        return np.nan
    return sum(remaining_flow) / len(remaining_flow)


def find_slope_from_minf_to_zero(t, flow, pef, t_offset=0):
    """
    In lieu of a compliance measurement now we will calculate the slope from min
    flow to 0.
    """
    min_idx = flow.index(pef)
    flow_min = (t[min_idx], min_idx, pef)
    flow_zero = (0, 0, sys.maxsize)  # (time, idx, flow)

    flow_threshold = 2
    for offset_idx, time in enumerate(t[flow_min[1]:]):
        idx = offset_idx + flow_min[1]
        if abs(flow[idx]) < flow_threshold and abs(flow[idx]) < flow_zero[2]:
            flow_zero = (time, idx, flow[idx])

    if flow_zero == (0, 0, sys.maxsize):
        return np.nan

    if (float(flow_zero[0]) - flow_min[0]) == 0:
        return np.nan

    slope = (float(flow_zero[2]) - flow_min[2]) / (float(flow_zero[0]) - flow_min[0])
    if slope < 0:
        return np.nan
    else:
        return slope


# 2015_06_22
def findx0(t, waveform, time_threshold):
    """
    Finds where waveform crosses 0 (changes from + to -)

    Args:
    t: time
    waveform: line to be analyzed
    time-threshold: upper limit (max value) for absolute value of time
    forward_dt: future point in waveform that must be negative

    Updated 2015/09/24 (SAM1.1.9) Stop evaluating if next value is nan
        (as in, non-data rows filled with 'nan' stop being considered as
         waveform[i+1])

    Updated 2015/09/11 and renamed SAM1_1_7; neg flow thresholds changed from
    <= -8 to <= -5 (note that 1st elif clause was found to be < -8 and was
    changed to be <= -5)

    Updated 2015/09/11 and renamed SAM1_1_6; included change in all clauses to
    replace <= to < signs. This dealt with run failures presumably due to
    trailing zeros at the end of the array. Also updated to include 0 in the
    definition of i to account for failures where the value just before the 1st
    neg value was 0 instead of positive.

    Updated 2015/09/09 and renamed SAM1_1_5 to signify SAM v1.1.5; included
    fourth elif clause to x0 logic to allow for cases in which flow 'dribbles'
    along at low values (e.g. -3) for a sustained period, never reaching -8
    threshold, but representing true exhalation event

    Updated 2015/09/04 2.2.6 Improve x0 function sensitivity with 3rd OR clause
        and smaller neg threshold
    Updated: 2015/09/03 2.2.4 Additional OR clause
    Updated: 2015/06/11
    Written: ?
    """
    t.extend([np.nan] * 6)
    waveform.extend([np.nan] * 6)
    cross0_time = []
    for i in range(len(waveform)-2): #if change to append scheme, will have to worry about -1 error
        if waveform[i]>=0 and waveform[i+1] is not np.nan:
            if waveform[i + 1] <= -5 and waveform[i + 2] < 0:
                    cross0_time.append(t[i + 1])
            elif waveform[i + 1]<0 and waveform[i + 4] <= -5:
                    cross0_time.append(t[i + 1])
            elif waveform[i+1]<0 and waveform[i+2]<=-5:
                    cross0_time.append(t[i + 1])
            elif waveform[i+1]<0 and waveform[i+2]<0 and waveform[i+3]<0 and \
                waveform[i+4]<0 and waveform[i+5]<0:
                    cross0_time.append(t[i + 1])

    i = 0
    while i <= len(cross0_time) - 2:
        if abs(cross0_time[i] - cross0_time[i + 1]) < time_threshold:
            del cross0_time[i + 1]
        else:
            i += 1

    for i in range(6):
        t.pop(-1)
        waveform.pop(-1)
    return cross0_time


def findx02(wave,dt):
    """
    Finds where waveform crosses 0 after largest portion contiguous positive AUC

    Args:
    wave: line to be analyzed (ex. flow)

    V1.0 2015-09-23 (2.0) SAM 1.1.8
    Find x02 separates the the wave into positive portions and negative portions.
    The largest positive portion will be considered the inspiratory portion.

    V1.1 2015-10-27 (2.1) SAM 1.2.0
    Utilizes AUC instead of just duration/length of portion

    20150615-V1.1 SAM 1.2.3 default for x0_index is []
    """
    posPortions=[] #holds all positive portion arrays
    negPortions=[] #holds all negative portion arrays
    hold=[] #holding array that is being built
    largestPos=0 #eventually becomes the largest pos AUC (tvi)
    largestNeg=0 #eventually becomes the largest neg AUC (tve)
    x0_index=[] #index where x0 occurs

    for i in range(len(wave)-1): #for each value in the wave
        if wave[i]>0: # if the value is greater than 0, it is considered positive
            hold.append(wave[i]) # and will be added to the holding array
            sign = 'pos'
        else: # if the value isn't greater than 0, it is considered negative
            hold.append(wave[i]) # and will be added to the holding array
            sign = 'neg'

        if wave[i+1]>0: #determine the sign of the next value in the wave
            nextSign = 'pos'
        else:
            nextSign = 'neg'

        if sign != nextSign: #if the sign is different than the sign of next value
            # save the holding array
            if sign=='pos':
                posPortions.append(hold)
                #calculate areas under the curve (tvi)
                holdAUC = simps(hold, dx=dt)*1000/60 #1000ml/L, 60 sec/min
                if holdAUC>largestPos: #if holding array has largest AUC
                    largestPos=holdAUC #it is now considered the largest AUC array
                    x0_index=i+1 #x0 will be considered time + 1
            if sign =='neg': # similar to positive
                negPortions.append(hold)
                holdAUC = simps(hold, dx=dt)*1000/60 #1000ml/L, 60 sec/min
                if holdAUC<largestNeg:
                    largestNeg=holdAUC
            hold=[]
            #possibly add some additional thing here?
    return posPortions, negPortions, largestPos, largestNeg, x0_index
#    return posPortions, negPortions, longestPos,longestNeg, x0_index

def calcTV3(wave,dt,x02index):
    """
    Written 2015/10/27
    """
    tvi=0
    tve=0
    hold=[] #holding array
    for i in range(len(wave)-1):#for each value in the wave
        if wave[i]>0: # if the value is greater than 0, it is considered positive
            hold.append(wave[i]) # and will be added to the holding array
            sign = 'pos'
        else: # if the value isn't greater than 0, it is considered negative
            hold.append(wave[i]) # and will be added to the holding array
            sign = 'neg'

        if wave[i+1]>0: #determine the sign of the next value in the wave
            nextSign = 'pos'
        else:
            nextSign = 'neg'

        if sign != nextSign: #if the sign is different than the sign of next value
            if i<x02index and sign=='pos':
                holdAUC = simps(hold, dx=dt)*1000/60 #1000ml/L, 60 sec/min
                tvi+=holdAUC
            elif i>=x02index and sign =='neg':
                holdAUC = simps(hold, dx=dt)*1000/60 #1000ml/L, 60 sec/min
                tve+=holdAUC
            else:
                pass

    return tvi, tve
def writecsv(outputM, OUTPUT_FILE):
    """writes csv using CSV reader, requires python 2.7"""
    with open(OUTPUT_FILE, 'wb') as outputopen:
        outputwriter = csv.writer(outputopen, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in outputM:
            outputwriter.writerow(row)

    #print "'" + OUTPUT_FILE + "' written in" + '\n\t' + os.getcwd()

def isFlat(data, epsilon = 1, y=0):
    """
    Determines if a region is flat around the horizontal line y.

    This function is used in the delayed trigger algorithms

    ARGS:
    data: 1D list or array (ex. e_pressure)
    epsilon: upper/lower bound
    y: value that the data approaches

    RETURNS:
    flatLengths: list containing lengths of regions that meet criteria
    maxFlat: longest length (units: index numbers, NOT time)
    sumFlat: sum of flatLenghts, another way of measuring time spent near y

    written: 2015/05/23
    """
    flatLengths = []
    k = 0
    for row in data:
        if abs(row-y)<epsilon:
            k+=1
        else:
            if k>0:
                flatLengths.append(k)
                k = 0
    if flatLengths !=[]:
        maxFlat = max(flatLengths)
        sumFlat = sum(flatLengths)
    else:
        maxFlat = 0
        sumFlat = 0

    return flatLengths, maxFlat, sumFlat


def find_x0s_multi_algorithms(flow, t, last_t, dt):
    """
    Calculate x0s based on multiple algorithms

    versions
    20160503 V1 Original, from TOR 3.5.1
    20160613 V1.1 Disregard ts (time stamp)
    20160721 V1.2 Change output to dictionary
    20160722 V2 Make only output indices
    """
    x0_indices_dict = {}

    #index #1
    x01s = findx0(t, flow, 0.5)

    if x01s!=[]: #if x01 has multiple values, use the first value to mark end of breath
        x01index=t.index(x01s[0])
    else:# if breath doesn't cross 0 (eg. double trigger, nubbin)
        x01index = t.index(last_t) #???perhaps we should set to beginning of breath?

    #index #2
    pos,neg,FlowLargePos,FlowLargeNeg,x02index = findx02(flow,dt)
    if x02index==[]:
        x02index = len(flow) - 1

    #save output
    x0_indices_dict['x01index'] = x01index
    x0_indices_dict['x02index'] = x02index

    return x0_indices_dict


def x0_heuristic(x0_indices_dict,BN,t):
    """
    Determine which x0 to use
    20160503 V1 Original, from TOR 3.5.1
    20160716 V2 Remove ts dependency
    """

    x01index=int(x0_indices_dict['x01index'])
    x02index=int(x0_indices_dict['x02index'])

    # THIS IS ESPECIALLY IMPORTANT IN NUBBIN BREATHS
    if x02index>x01index:
        x0_index=x02index
        iTime=t[x02index]
    else:
        iTime=t[x01index]
        x0_index=x01index

    return iTime,x0_index

#! /usr/bin/python3
# 12-3-2021
# The main routine for looping many GNSS ENU time series for 3 components

# Function_95CI.py, Main_cal_95CI.py, and input *.col" files should be in the same directory
# For running the program, just type "./Main_cal_95CI.py 

import os
import pandas as pd
from GNSS_Vel_95CI import cal_95CI

directory = './'
# An example of 'fin': UH01_GOM20_neu_cm.col
# Decimal-Year      NS(cm)       EW(cm)       UD(cm)  sigma-NS(cm)  sigma-EW(cm)  sigma-UD(cm)
#   2012.7447       0.0501       0.1444       0.3072       0.0495       0.0217       0.0541
#   2012.7474       0.0336      -0.1013      -0.3132       0.0485       0.0208       0.0527
#   2012.7502      -0.1226       0.0234       0.3309       0.0499       0.0215       0.0544
#   2012.7529      -0.0505      -0.0609      -0.4559       0.0487       0.0217       0.0534
#   ......

for fin in os.listdir(directory):
    if fin.endswith("neu_cm.col"):
       print(fin)
       GNSS = fin[0:4]    # station name, e.g., UH01
       ts_enu = []
       ts_enu = pd.read_csv (fin, header=0, delim_whitespace=True)
       year = ts_enu.iloc[:,0]    # decimal year
    
       dis = ts_enu.iloc[:,1]     # NS
       result_NS=cal_95CI(year,dis,GNSS, ENU='NS',plot='on', pltshow='on')
       b_NS=round(result_NS[0],2)          # slope, or site velocity
       b_NS_95CI=round(result_NS[1],2)      # The 95%CI of slope
 
       dis = ts_enu.iloc[:,2]     # EW
       result_EW=cal_95CI(year,dis,GNSS,ENU='EW',plot='on',pltshow='on')
       b_EW=round(result_EW[0],2)
       b_EW_95CI=round(result_EW[1],2)
        
       dis = ts_enu.iloc[:,3]     # UD
       result_UD=cal_95CI(year,dis,GNSS,ENU='UD',plot='on',pltshow='on')
       b_UD=round(result_UD[0],2)
       b_UD_95CI=round(result_UD[1],2)
       
    else:
       continue
       



#! /usr/bin/python3
# 11-29-2021
# The main routine for looping many GNSS ENU time series for 3 components

# Function_95CI.py, Main_cal_95CI.py, and input *.col" files should be in the same directory
# For running the program, just type "./Main_cal_95CI.py 

import os
import pandas as pd
from pandas import read_csv
# from Function_GNSS_95CI import cal_95CI
from GNSS_Vel_95CI import cal_95CI

# directory = '/home/gwang/Python_Program/cal_95CI'

directory = './'
# The 'fin' file should include 4 or more columns
# col1:"Decimal_Year'; col2:"Dis.NS"; Col3:"Dis.EW"; Col4:"Dis.UD"

for fin in os.listdir(directory):
    if fin.endswith("neu_cm.col"):
  #  if fin.endswith("XP5_HRF20_neu_cm.col"):
       print(fin)
       # Extract two columns from the input file: decimal-year and displacement
       GNSS = fin[0:4]    # station name, e.g., UH01
       plot ='on'         # or 'off'
       
       ts_enu = []
       ts_enu = pd.read_csv (fin, header=0, delim_whitespace=True)
       
       year = ts_enu.iloc[:,0]   # decimal year
       
       # Loop three components(NS, EW, UD) one by one    
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
       



# GNSS_Vel_95-CI
A Python program for calculating the 95%CI for GNSS-derived site velocities

Function_GNSS_95CI.py is a Python function for calculating the site velocity (b) and its 95%CI for GNSS-derived daily displacement time series.
The detailed methods are adressed in:

Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390

Main_cal_95CI.py illustrates the method of implementing "Function_GNSS_95CI.py" into your own Python program.
You may need to install "pandas", "matplotlib", "statsmodels" on yur computer if you have not used them before.
How--- "pip install pandas", "pip install matplotlib", "pip install statsmodels".

How to test the package on your computer?
Put following files on a folder:
Main_cal_95CI.py
Function_GNSS_95CI.py
UH01_GOM20_neu_cm.col
.....

Change your work directory to the folder,
For Linux system users:
Type "./Main_cal_95CI.py" in your terminal
For Windows system users:
Type " python Main_cal_95CI.py" in your CMD window

# result_NS=cal_95CI(year,dis,GNSS, ENU='NS',plot='on', pltshow='on')
Try plot='off', pltshow='off'

Good Luck!
bob.g.wang@gmail.com

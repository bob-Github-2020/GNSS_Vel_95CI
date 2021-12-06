# GNSS_Vel_95%CI
A Python program for calculating the 95%CI for GNSS-derived site velocities

GNSS_Vel_95CI.py is a Python module for the calculation of the site velocity (b) and its 95%CI for GNSS-derived daily displacement time series.
The detailed methods are adressed in:
Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390

Main_cal_95CI.py illustrates the method of implementing "GNSS_Vel_95CI.py" into your own Python program.
You may need to install "pandas", "matplotlib", "statsmodels" on yur computer if you have not used them before.

How--- "pip install pandas", "pip install matplotlib", "pip install statsmodels".

How to run the module on your computer?
Download following files into a folder:

Main_cal_95CI.py

GNSS_Vel_95%CI.py

UH01_GOM20_neu_cm.col  (sample file)
.....

Change your work directory to the folder,

For Linux system users:

Type "./Main_cal_95CI.py" in your terminal

For Windows system users:

Type " python Main_cal_95CI.py" in your CMD window

Good Luck!

bob.g.wang@gmail.com

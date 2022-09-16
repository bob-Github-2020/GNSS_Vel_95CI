# GNSS_Vel_95%CI
A Python program for calculating the 95%CI for GNSS-derived site velocities, by B. Cornelison and G. Wang.

GNSS_Vel_95CI.py is a Python module for the calculation of the site velocity (b) and its 95%CI for GNSS-derived daily displacement time series.
The detailed methods are adressed in:

Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390

Cornelison and Wang (2023). GNSS_Vel_95CI.py: A Python Module for Calculating the Uncertainty of GNSS-Derived Site Velocity, J. Surv. Eng. 2022, 149(1): 06022001. http://doi.org/10.1061/(ASCE)SU.1943-5428.0000410

The source codes are avaliable at pip and GitHub:

https://pypi.org/project/GNSS-Vel-95CI

https://github.com/bob-Github-2020/GNSS_Vel_95CI

You may install the module on your computer by "pip install GNSS_Vel_95CI" or download the source code (GNSS_Vel_95CI.py) from GitHub and put it into your working directory.

Main_cal_95CI.py illustrates the method of implementing "GNSS_Vel_95CI.py" into your own Python program.

You may need to install "pandas", "matplotlib", "statsmodels" on yur computer if you have not used them before.

How--- "pip install pandas", "pip install matplotlib", "pip install statsmodels".


# How to run the module on your computer? 

Download following files into a folder:

    Main_cal_95CI.py

    GNSS_Vel_95%CI.py (You donot need this one if you already installed the moduler on your computer by pip)

    MRHK_GOM20_neu_cm.col  (sample file)

    .....

Change your work directory to the folder:

For Linux system users:

    Type "./Main_cal_95CI.py" in your terminal

For Windows system users:

    Type " python Main_cal_95CI.py" in your CMD window


Good Luck!

For comments, please contact:

bob.g.wang@gmail.com


# Two figures output from the module

![MRHK_UD_ACF](https://user-images.githubusercontent.com/65426380/167514723-83626229-3c40-4256-8bbc-f22d2082bd98.png)

![MRHK_UD_Decomposition](https://user-images.githubusercontent.com/65426380/181590972-d1e231c7-b95f-499a-836f-0e9c0dee0903.png)



# Detailed Method

[Methods_Vel_95CI.pdf](https://github.com/bob-Github-2020/GNSS_Vel_95CI/files/7664316/Methods_Vel_95CI.pdf)

[95CI_Python.pdf](https://github.com/bob-Github-2020/GNSS_Vel_95CI/files/9590450/95CI_Python.pdf)

# GNSS_Vel_95CI.py

Last updated: 2-2-2023, by Guoquan Wang, bob.g.wang@gmail.com

GNSS_Vel_95CI.py is a Python module for the calculation of the site velocity (b) and its 95%CI for GNSS-derived daily displacement time series.
The detailed methods are adressed in:

Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng., 148(1): 04021030. 
http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390

Cornelison and Wang (2023). GNSS_Vel_95CI.py: A Python Module for Calculating the Uncertainty of GNSS-Derived Site Velocity, J. Surv. Eng., 149(1): 06022001. http://doi.org/10.1061/(ASCE)SU.1943-5428.0000410

The module has been archived at pypi.org: 

     https://pypi.org/project/GNSS-Vel-95CI
     
You can install the Python module on your computer by:
  
     pip install GNSS-Vel-95CI

The source code, an example main routine (Main_cal_95CI.py), and sample datasets are archived at my Github site: 

     https://github.com/bob-Github-2020/GNSS_Vel_95CI
 
Main_cal_95CI.py illustrates the method of calling "GNSS_Vel_95CI.py" in your own Python program.

You may need to install "pandas", "matplotlib", "statsmodels" on yur computer if you have not used them before.

      pip install module-name


# How to run the module on your computer? 

Download following files into a folder:

    Main_cal_95CI.py

    GNSS_Vel_95CI.py (You donot need this one if you already installed the moduler on your computer by " pip install ")

    MRHK_GOM20_neu_cm.col  (sample file)

    .....

Change your work directory to the folder:

For Linux system users, type the following command in your terminal:

    ./Main_cal_95CI.py

For Windows system users, type the following command in your CMD terminal:

    python Main_cal_95CI.py (or py Main_cal_95CI.py, python3 Main_cal_95CI.py)


Good Luck!


# Two example figures output from the module

![MRHK_UD_ACF](https://user-images.githubusercontent.com/65426380/167514723-83626229-3c40-4256-8bbc-f22d2082bd98.png)

![MRHK_UD_Decomposition](https://user-images.githubusercontent.com/65426380/181590972-d1e231c7-b95f-499a-836f-0e9c0dee0903.png)


# Detailed Method

[Methods_Vel_95CI.pdf](https://github.com/bob-Github-2020/GNSS_Vel_95CI/files/7664316/Methods_Vel_95CI.pdf)

[95CI_Python.pdf](https://github.com/bob-Github-2020/GNSS_Vel_95CI/files/9590450/95CI_Python.pdf)

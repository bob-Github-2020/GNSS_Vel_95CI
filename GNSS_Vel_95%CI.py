#!/usr/bin/python3
# Last updated: 12-1-2021 
# The function for calculating site velocity (b) and its 95%CI
# The detailed methods are adressed in:
#    Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
#    http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390

import os
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pandas import read_csv
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
# from sklearn.impute import KNNImputer

def GNSS_Vel_95%CI(year,ts,GNSS,ENU,plot,pltshow):
    """
    Calculate the slope (site velocity) of GNSS-Displacement time series and its 95%CI
    
    Inputs
    -------
    year: decimal year, e.g., 2021.003 ...
    ts: displacement time series, in cm or mm
    GNSS: string of GNSS station name, e.g., UH01
    ENU: string, NS or EW or UD component
    plot: 'on' or 'off'. 'on' will plot figures (*.pdf) and write out data (*.txt).
    pltshow: 'on' or 'off'. 'on' will display the figures on the screen. You need to close the figure to continue other files
         
    Returns
    --------
    b_L, b95CI
    
   
    Output Figures and Files
    ------------------------
           if plot='on', produce 2 figures and 3 files
           Figures: ACF, Decomposition
           Files: ACF time series, Non-Linear trend, Seasonal Model
           
    
    Preparing the input file 'fin'
    ------------------------------
    Preparing input GNSS daily ENU time series (fin) as 4 columns or more
    Decimal-Year  NS(cm)  EW(cm)  UD(cm)
    An example of 'fin': UH01_GOM20_neu_cm.col
    Decimal-Year    NS(cm)       EW(cm)       UD(cm) sigma-NS(cm) sigma-EW(cm) sigma-UD(cm)
    2012.745        0.050        0.144        0.307        0.050        0.022        0.054
    2012.747        0.034       -0.101       -0.313        0.048        0.021        0.053
    2012.750       -0.123        0.023        0.331        0.050        0.022        0.054
    2012.753       -0.050       -0.061       -0.456        0.049        0.022        0.053

    An examples for implementing the function in your Python program
    ----------------------------------------------------------------
    #!/usr/bin/python3
    import os
    import pandas as pd
    from pandas import read_csv
    from Function_GNSS_95CI import cal_95CI

    directory = './'
    # copy all of your data files to the same directory with the function
    
    for fin in os.listdir(directory):
    if fin.endswith("neu_cm.col"):     # adjust with your file names
    
       GNSS = fin[0:4]    # station name, e.g., UH01
       ts_enu = []
       ts_enu = pd.read_csv (fin, header=0, delim_whitespace=True)
       year = ts_enu.iloc[:,0]   # decimal year
       
       # Loop three components(NS, EW, UD) one by one    
       dis = ts_enu.iloc[:,1]     # NS
       result_NS=cal_95CI(year,dis,GNSS, ENU='NS',plot='on')
       b_NS=round(result_NS[0],2)          # slope, or site velocity
       b_NS_95CI=round(result_NS[1],2)      # The 95%CI of slope
      
       dis = ts_enu.iloc[:,2]     # EW
       result_EW=cal_95CI(year,dis,GNSS,ENU='EW',plot='on')
       b_EW=round(result_EW[0],2)
       b_EW_95CI=round(result_EW[1],2)
        
       dis = ts_enu.iloc[:,3]     # UD
       result_UD=cal_95CI(year,dis,GNSS,ENU='UD',plot='on')
       b_UD=round(result_UD[0],2)
       b_UD_95CI=round(result_UD[1],2)
       
    else:
       continue  
    # End of the Main program
       
    Reference
    ---------
    The detailed methods are adressed in:
    Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
    http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390
 
    """
    
    N=len(ts)    # Total points
    T=year[N-1]-year[0]   # Total years
   
# -----------------------------------------------------------------------------
# Step 1: Linear regresion on the whole time series
#         Eq.1: Li=a+b*ti+Ri, using OLS--Ordinary Least Squares
# -----------------------------------------------------------------------------
    x = sm.add_constant(year)
    model = sm.OLS(ts,x)
    results = model.fit()
    b_L = results.params[1]
   
    # stand error. SEs, Eq. 7
    s=np.sqrt(np.sum(results.resid**2)/results.df_resid)    # Eq.6
    SEs= s/np.sqrt(N)                                       # Eq.7
    SEb=SEs*2*np.sqrt(3.0)/T                                # Eq.8

    Li = results.params[0]+results.params[1]*year
# -----------------------------------------------------------------------------
# Step 2: Calculate the slope (b_NL) of the non-linear component (NLi)
#         The non-linear trend is obtained from LOWESS filter
#         yi=Li+NLi+Si+ri, Eq.9 
# -----------------------------------------------------------------------------
    Ri = ts - Li
    # smooth Ri with LOWESS
    x_tmp = np.array(year)
    y_tmp = np.array(Ri)

    Ri_smooth = sm.nonparametric.lowess(y_tmp, x_tmp, frac= 1./3, it=2)
    NLi=Ri_smooth[:,1]

    # cal Linear trend of NL(i)
    x = sm.add_constant(x_tmp)
    model = sm.OLS(NLi,x)
    results = model.fit()
    NLi_line=results.params[0]+results.params[1]*year
    b_NL = results.params[1]

# -----------------------------------------------------------------------------
# Step 3: Setup the seasonal model (Si), calculate b_S
#         The data gap needs to be filled 
# -----------------------------------------------------------------------------
    res_L_NL = Ri-NLi
    
    def decimalYear2Date(dyear):
        year = int(dyear)
        yearFraction = float(dyear) - year
        doy = int(round(yearFraction * 365.25-0.5)) + 1
        ydoy = str(year) + "-" + str(doy)
        r = datetime.strptime(ydoy, "%Y-%j").strftime("%Y-%m-%d")
        return r  

    # Preparing for filling gaps
    # use a loop converting original decimal year to date, e.g., 2021-05-25
    ymdR = []
    for line  in year:
        ymdi = decimalYear2Date(line)
        ymdR.append(ymdi)
    
    # convert row to column
    ymd = pd.DataFrame (ymdR)

    # combine column ymd and res_L_NL
    ymd_and_res = pd.concat([ymd, res_L_NL], axis=1)

    # add column name to the DataFrame
    ymd_and_res.columns = ['Date', 'RES']
    df = ymd_and_res

    # Convert column "Date" to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')

    # Firstly, fill the gap in YMD seris and give NaN for RES series
    df_con_nan = df.resample('1D').mean()      # 1D---1day
    y_con_nan=df_con_nan['RES']    # used for output
    y_con_nan=y_con_nan.reset_index()

    # Secondly, fill the NaN in RES column as a number, use assign, or random, prefer random
    # df_con = df_con_nan['RES'].interpolate(method='linear')  # This works
    # df_con = df_con_nan.assign(InterpolateTime=df_con_nan.RES.interpolate(method='time'))   # This also works

    def fill_with_random(df2, column):
        '''Fill df2's column  with random data based on non-NaN data from the same column'''
        df = df2.copy()
        df[column] = df[column].apply(lambda x: np.random.choice(df[column].dropna().values) if np.isnan(x) else x)
        return df
    
    df = fill_with_random(df_con_nan,'RES')

    # Calculate Seasonal coefficients, see Eq.10
    # df include "2012-12-14   -0.087698". The first col is index. 
    df = df.reset_index()
    df = pd.DataFrame(df)
    x_con = df.iloc[:,0]
    y_con = df.iloc[:,1]

    # Build continuous decimal year time series, xt
    x0 = year[0]
    npts = len(y_con) 
    xt=np.zeros(npts)
    for i in range(npts):
        xt[i] = x0 + i*1/365.25
      
    # The function for calculating Seasonal Model coeffients
    def seasonal_mod(x,y):
        twopi = 2.0 * np.pi
        x0=x[0]
        x=x-x0+1.0/365.25
       
        # For this method, just use integer Years data, e.g., 10 years not 10.3
        npoint_in=len(y)
        ny = int(np.floor(npoint_in/365.25))
        npts = int(ny*365.25)   # used points of ny years
        dy = 1.0/365.25
        rn = 1.0/npts
    
        # mp--maximum ip for an, bn should be 3 times or larger
        mp = int(3*ny)
        a=np.zeros(mp)
        b=np.zeros(mp)
    
        for ip in range(mp):
            a[ip]=0
            b[ip]=0

            for i in range(npts):
                a[ip]=a[ip]+2.0*rn*y[i]*np.cos(twopi*(ip-1)*i*rn)
                b[ip]=b[ip]+2.0*rn*y[i]*np.sin(twopi*(ip-1)*i*rn)
           
        a0=a[1]
        a1=a[ny+1]
        b1=b[ny+1]
        a2=a[2*ny+1]
        b2=b[2*ny+1]
        
        Si=a0+a1*np.cos(1.0*twopi*x)+b1*np.sin(1.0*twopi*x)+a2*np.cos(2.0*twopi*x)+b2*np.sin(2.0*twopi*x) 
        
        return Si, a0, a1, b1, a2, b2

    result_seasonM= seasonal_mod(xt,y_con)
    
    Si=result_seasonM[0]
    
    # output a0,a1,b1,a2,b2 for plotting on the final figure
    a0=result_seasonM[1]
    a1=result_seasonM[2]
    b1=result_seasonM[3]
    a2=result_seasonM[4]
    b2=result_seasonM[5]

    # calculate the linear trend of Si
    x = sm.add_constant(xt)
    model = sm.OLS(Si,x)
    results = model.fit()
    Si_Line=results.params[0]+results.params[1]*xt
    b_S = results.params[1]

# -----------------------------------------------------------------------------
# Step 4: calculate the Effect Sample Size, Neff--Eq.13, and SEbc--Eq.15
#         work on ri, yi=Li+NLi+Si+ri, Eq.9 
# -----------------------------------------------------------------------------
    ri = y_con - Si
    
    # cal RMS of ri
    RMS_ri= math.sqrt(np.square(ri).mean())

    # get ACF and PACF, cal PACF is very slow. Doesnot need PACF! 
    data = np.array(ri)
    lag_acf = acf(data, nlags=1000,fft=True)
    # lag_pacf = pacf(data, nlags=1000, method='ols')
    
    ## Plot and write out ACF
    if plot == 'on':
       y = pd.DataFrame(lag_acf)
       y.to_csv(GNSS +'_'+ ENU +'_ACF.txt', index = True, header=False)
    
       # Plot ACF
       if len(data) < 1000:
          nlag = len(data)-1
       else:
          nlag=1000 
          
       plot_acf(data, fft=True, lags=nlag,zero=False, title='ACF: '+ GNSS+'_'+ENU)
       plt.savefig(GNSS + ENU + "_ACF.pdf")
       # plt.show()
    # end if
       
    sum = 0
    i=0
    for acfi in lag_acf:
        if acfi >= 0:
           i=i+1
           sum = sum + acfi
        else:
            # print("Found lag-M at", i)
            break

    tao = 1 + 2*sum            # Eq.14
    Neff = N/tao               # Eq.13
    SEbc=np.sqrt(tao)*SEb      # Eq.15

# -----------------------------------------------------------------------------
# Step 5: calculate the 95%CI--Eq.16, and predicted 95%CI--Eq.17 and Eq.18
# -----------------------------------------------------------------------------

    b95CI = 2 * SEbc + abs(b_NL) + abs(b_S)     #Eq.16
  
    # cal the predicted 95%CI (mm/year) based on the Formulas Eq.17 and Eq.18
    if ENU == 'UD':
       b95CI_mod = 5.2/math.pow(T,1.25)
    else:
       b95CI_mod = 1.8/T
       
# -----------------------------------------------------------------------------
# Step 6: Plot the decomposition components, Fig.3, if plot = on
# -----------------------------------------------------------------------------
    if plot == 'on':
       fig, (fig1,fig2,fig3,fig4) = plt.subplots(4, figsize=(16,14))
       fig.subplots_adjust(hspace=0.3)
       fig.suptitle('Decomposition of GNSS-Derived Daily Displacement Time Series: '+ GNSS + '-' + ENU, size=16,  y=0.93);
    
       fig1.plot(year, ts, 'k.')
       fig1.plot(year,Li, 'r.')
       str_bL=str(round(b_L*10,2))
       str_b95CI=str(round(b95CI*10,2))
       str_b95CI_mod=str(round(b95CI_mod,2))
       str_a0=str(round(a0,2))
    
       if a1 >= 0:
          str_a1='+'+str(round(a1,2))
       else:
          str_a1=str(round(a1,2))
       
       if b1 >= 0:
          str_b1='+'+str(round(b1,2))
       else:
          str_b1=str(round(b1,2))
       
       if a2 >= 0:
          str_a2='+'+str(round(a2,2))
       else:
          str_a2=str(round(a2,2))
       
       if b2 >= 0:
          str_b2='+'+str(round(b2,2))
       else:
          str_b2=str(round(b2,2))
           
       fig1.text(0.15, 0.07, 'Site velocity: '+ str_bL + '$\pm$' + str_b95CI+' mm/year', ha='center', va='center', transform=fig1.transAxes)
       fig1.text(0.7, 0.07, 'Calculated vs. Predicted 95%CI: '+ str_b95CI + ' vs. '+ str_b95CI_mod + ' mm/year', ha='center', va='center', transform=fig1.transAxes)
     
       fig3.text(0.2,0.07, 'S='+str_a0+str_a1+'cos(2$\pi$x)' + str_b1+'sin(2$\pi$x)'+str_a2+'cos(4$\pi$x)'+str_b2+'sin(4$\pi$x)', ha='center', va='center', transform=fig3.transAxes)
      
       str_RMS_ri=str(round(RMS_ri*10,1))
       fig4.text(0.1, 0.1, 'RMS: '+ str_RMS_ri + ' mm', ha='center', va='center', transform=fig4.transAxes)

       fig2.plot(year, Ri,'c.')
       fig2.plot(year, NLi, 'r.')

       fig3.plot(xt, y_con,'c.')
       fig3.plot(xt, Si,'r.')

       fig4.plot(xt, ri,'r.')

       fig1.set_ylabel('Dis. (cm)')
       fig2.set_ylabel('Dis. (cm)')
       fig3.set_ylabel('Dis. (cm)')
       fig4.set_ylabel('Dis. (cm)')

       fig4.set_xlabel('Year')

       fig1.set_title('(a) Displacement y(i) and Linear Trend L(i)')
       fig2.set_title('(b) Non-Linear Trend NL(i)')
       fig3.set_title('(c) Seasonal Model S(i)')
       fig4.set_title('(d) Residues r(i)')

       fig.savefig(GNSS + '_' + ENU + '_Decomposition.png')
       fig.savefig(GNSS + '_' + ENU + '_Decomposition.pdf')
       
       if pltshow == 'on':
          plt.show()

       # output the time series, original and filled
       f1_out = GNSS + "_" + ENU + "_Linear_NonLinear.txt"
       # build the DataFrame
       NLi=pd.DataFrame(NLi)
       df = pd.concat([year, ts*10, Li*10, Ri*10, NLi*10], axis=1)
       # add column name to the DataFrame
       df.columns = ['Year', 'Dis(mm)','Linear','Residue','Smoothed']
       df.to_csv(f1_out, header=True, index=None, sep=' ', mode='w', float_format='%.5f')

       xt=pd.DataFrame(xt)
       Si=pd.DataFrame(Si)
       y_con=pd.DataFrame(y_con)
       ri=pd.DataFrame(ri)
       f2_out = GNSS + "_" + ENU + "_SeasonalM.txt"
       df = pd.concat([xt, y_con*10,Si*10,ri*10], axis=1)
       df.columns = ['Year_con', 'Dis_filled', 'SeasonMod','Final_Res']
       df.to_csv(f2_out, header=True, index=None, sep=' ', mode='w', float_format='%.5f')
       
    else: 
       print('Plot is off!')  

    return b_L, b95CI
 







    
  




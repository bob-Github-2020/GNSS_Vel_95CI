#! /usr/bin/python3
# Last updated: 01-06-2023, by G. Wang
# The function for calculating site velocity (b) and its 95%CI
# You may install the module on your computer by: pip install GNSS_Vel_95CI
# https://pypi.org/project/GNSS-Vel-95CI/
# https://github.com/bob-Github-2020/GNSS_Vel_95CI
# The detailed methods are adressed in:
#    Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
#         http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390
#    Cornelison B. and Wang G.(2023). GNSS_Vel_95CI.py: A Python Module for Calculating the Uncertainty of GNSS-Derived 
#         Site Velocity. J. Surv. Eng. 2022, 149(1)xxx. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000410


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import acf

plt.rcParams.update({'font.size': 14})

def cal_95CI(year,ts,GNSS,DIR,output,pltshow):

    """
    Calculate the slope (site velocity) of GNSS-Displacement time series and its 95%CI

    Inputs
    -------
    year: decimal year, e.g., 2021.003 ...
    ts: displacement time series, in cm or mm
    GNSS: string, GNSS station name, e.g., UH01
    DIR: string, NS or EW or UD component
    output: 'on' or 'off'. 'on' will plot figures (*.png, *.pdf) and write out data (*.txt).
    pltshow: 'on' or 'off'. 'on' will display the figures on the screen. You need to close the figure to continue other files
         
    Returns
    --------
    b_L, b95CI
    
    Preparing the input file 'fin'
    ------------------------------
    Preparing input GNSS daily ENU time series (fin) as 4 columns or more
    Decimal-Year  NS(cm)  EW(cm)  UD(cm)
    An example of 'fin': UH01_GOM20_neu_cm.col
    # Decimal-Year      NS(cm)       EW(cm)       UD(cm)  sigma-NS(cm)  sigma-EW(cm)  sigma-UD(cm)
    #   2012.7447       0.0501       0.1444       0.3072       0.0495       0.0217       0.0541
    #   2012.7474       0.0336      -0.1013      -0.3132       0.0485       0.0208       0.0527
    #   2012.7502      -0.1226       0.0234       0.3309       0.0499       0.0215       0.0544
    #   2012.7529      -0.0505      -0.0609      -0.4559       0.0487       0.0217       0.0534
    #   ......

    An examples for implementing the function in your Python program
    ----------------------------------------------------------------
    #! /usr/bin/python3
    import os
    import pandas as pd
    from GNSS_Vel_95CI import cal_95CI

    directory = './'
           
    for fin in os.listdir(directory):
       if fin.endswith("neu_cm.col"):
          print(fin)
          GNSS = fin[0:4]    # station name, e.g., UH01
          ts_enu = []
          ts_enu = pd.read_csv (fin, header=0, delim_whitespace=True)
          year = ts_enu.iloc[:,0]    # decimal year
    
          dis = ts_enu.iloc[:,1]     # NS
          result_NS=cal_95CI(year,dis,GNSS,DIR='NS',output='on', pltshow='on')
          b_NS=round(result_NS[0],2)          # slope, or site velocity
          b_NS_95CI=round(result_NS[1],2)      # The 95%CI of slope
 
          dis = ts_enu.iloc[:,2]     # EW
          result_EW=cal_95CI(year,dis,GNSS,DIR='EW',output='on',pltshow='on')
          b_EW=round(result_EW[0],2)
          b_EW_95CI=round(result_EW[1],2)
        
          dis = ts_enu.iloc[:,3]     # UD
          result_UD=cal_95CI(year,dis,GNSS,DIR='UD',output='on',pltshow='on')
          b_UD=round(result_UD[0],2)
          b_UD_95CI=round(result_UD[1],2)
       
       else:
          continue

    Reference
    ---------
    The detailed methods are adressed in:
    Wang, G. (2022). The 95% Confidence Interval for GNSS-Derived Site Velocities, J. Surv. Eng. 2022, 148(1): 04021030. 
    http://doi.org/10.1061/(ASCE)SU.1943-5428.0000390
 
    """
    
    N=len(ts)             # Total points
    T=year[N-1]-year[0]   # Total year range
   
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
    # cal RMS of Ri, for printing on final figure, sub-Fig.2
    RMS_rm_L= math.sqrt(np.square(Ri).mean())
    
    # smooth Ri with LOWESS
    x_tmp = np.array(year)
    y_tmp = np.array(Ri)
    Ri_smooth = sm.nonparametric.lowess(y_tmp, x_tmp, frac= 1.0/2.5, it=2)
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
    # cal RMS of res_L_NL, for printing on final figure, sub-Fig.3
    RMS_rm_LNL= math.sqrt(np.square(res_L_NL).mean())
    
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
    def seasonal_model(x,y):
        twopi = 2.0 * np.pi
        x0=x[0]
        x=x-x0+1.0/365.25
       
        # For this method, just use integer Years of data, e.g., 10 years not 10.3
        npoint_in=len(y)
        ny = int(np.floor(npoint_in/365.25))
        npts = int(ny*365.25)   # used points of ny years
        dy = 1.0/365.25
        rn = 1.0/npts
    
        # mp--maximum ip should be 3 times ny or larger
        mp = int(3*ny)
        c=np.zeros(mp)
        d=np.zeros(mp)
    
        for ip in range(mp):
            c[ip]=0
            d[ip]=0
            for i in range(npts):
                c[ip]=c[ip]+2.0*rn*y[i]*np.cos(twopi*(ip-1)*i*rn)
                d[ip]=d[ip]+2.0*rn*y[i]*np.sin(twopi*(ip-1)*i*rn)
           
        c0=c[1]
        c1=c[ny+1]
        d1=d[ny+1]
        c2=c[2*ny+1]
        d2=d[2*ny+1]
        
        Si=c0+c1*np.cos(1.0*twopi*x)+d1*np.sin(1.0*twopi*x)+c2*np.cos(2.0*twopi*x)+d2*np.sin(2.0*twopi*x) 

        return Si, c0, c1, d1, c2, d2

    result_seasonM= seasonal_model(xt,y_con)
    Si=result_seasonM[0]
    
    # output c0,c1,d1,c2,d2 for plotting on the final figure
    c0=result_seasonM[1]
    c1=result_seasonM[2]
    d1=result_seasonM[3]
    c2=result_seasonM[4]
    d2=result_seasonM[5]

    # calculate the linear trend of Si
    x = sm.add_constant(xt)
    model = sm.OLS(Si,x)
    results = model.fit()
    Si_Line=results.params[0]+results.params[1]*xt
    b_S = results.params[1]
    
    # cal annual and hal-annual amplitudes,P2T--Peak to trough amplitude 
    P1=math.sqrt(np.square(c1)+np.square(d1))
    P2=math.sqrt(np.square(c2)+np.square(d2))
    P2T=math.sqrt(np.square(P1)+np.square(P2))*2.0
# -----------------------------------------------------------------------------
# Step 4: calculate the Effect Sample Size, Neff--Eq.13, and SEbc--Eq.15
#         work on ri, yi=Li+NLi+Si+ri, Eq.9 
# -----------------------------------------------------------------------------
    ri = y_con - Si
    
    # cal RMS of ri
    RMS_ri= math.sqrt(np.square(ri).mean())

    # get ACF and PACF, cal PACF is very slow. Doesnot need PACF!
    # Plot ACF
    if len(ri) < 1095:
       maxlag = len(ri)-1
    else:
       maxlag=1095 

    data = np.array(ri)
    lag_acf = acf(data, nlags=maxlag,fft=True)
    # lag_pacf = pacf(data, nlags=1000, method='ols')
          
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
    Neff = int(N/tao)          # Eq.13
    SEbc=np.sqrt(tao)*SEb      # Eq.15, same as SEbc=np.sqrt(N/Neff)*SEb
    
# -----------------------------------------------------------------------------
# Step 5: calculate the 95%CI--Eq.16, and projected 95%CI--Eq.17 and Eq.18
# -----------------------------------------------------------------------------
    b95CI = 1.96 * SEbc + abs(b_NL) + abs(b_S)     #Eq.16
  
    # cal the predicted 95%CI (mm/year) based on the Formulas Eq.17 and Eq.18
    if DIR == 'UD':
       b95CI_mod = 5.2/math.pow(T,1.25)
    else:
       b95CI_mod = 1.8/T     
# -----------------------------------------------------------------------------
# Step 6: Plot the decomposition components, Fig.3, if output = on
# -----------------------------------------------------------------------------
    if output == 'on':
       # write ACF
       y = pd.DataFrame(lag_acf)
       y.to_csv(GNSS +'_'+ DIR  + '_ACF.txt', index = True, header=True)
                   
       #plot_acf(data, fft=True, lags=nlag,zero=False, bartlett_confint=False, auto_ylims=True, adjusted=False,title='ACF: '+ GNSS+'_'+DIR)
       x=np.arange(0,len(y),1)
       x=np.array(x)
       y=np.array(y)
       plt.plot(x,y,'k.',markersize=2)
       y2=y[2]*1.1
       plt.ylim(top=y2)
       plt.xlim(right=maxlag)

       y=y.ravel()
       plt.fill_between(x, y)
       plt.xlabel('Time-lag (Days)')
       plt.ylabel('ACF')
       plt.title('ACF: '+GNSS+'('+ DIR +')')

       plt.savefig(GNSS +'_'+ DIR + "_ACF.pdf")
       plt.savefig(GNSS +'_'+ DIR + "_ACF.png"
       
       # Plot decompositions
       fig, (fig1,fig2,fig3,fig4) = plt.subplots(4, figsize=(16,14))
       fig.subplots_adjust(hspace=0.4)
       fig.suptitle('Decomposition of GNSS-Derived Daily Displacement Time Series: '+ GNSS + '(' + DIR + ')', size=16,  y=0.93);
       
       
       fig1.plot(year, ts, 'k.')
       fig1.plot(year,Li, 'r.')
       fig1.set_ylim(bottom=min(ts)*1.2, top=max(ts)*1.2)
      
       str_bL=str(round(b_L*10,2))
       str_bNL=str(round(b_NL*10,2))
       str_bS=str(round(b_S*10,2))
       str_b95CI=str(round(b95CI*10,2))
       str_b95CI_mod=str(round(b95CI_mod,2))   # mm/year
       str_c0=str(round(c0,2))
       str_SEb=str(round(SEb*10,2))
       str_SEbc=str(round(SEbc*10,2))
       
       str_RMS_rm_L=str(round(RMS_rm_L*10,1))
       str_RMS_rm_LNL=str(round(RMS_rm_LNL*10,1))
       str_RMS_ri=str(round(RMS_ri*10,1))
       
       str_P1=str(round(P1*10,1))
       str_P2=str(round(P2*10,1))
       str_P2T=str(round(P2T*10,1))

       if c1 >= 0:
          str_c1='+'+str(round(c1,2))
       else:
          str_c1=str(round(c1,2))
       
       if d1 >= 0:
          str_d1='+'+str(round(d1,2))
       else:
          str_d1=str(round(d1,2))
       
       if c2 >= 0:
          str_c2='+'+str(round(c2,2))
       else:
          str_c2=str(round(c2,2))
       
       if d2 >= 0:
          str_d2='+'+str(round(d2,2))
       else:
          str_d2=str(round(d2,2))
           
       fig1.text(0.5, 0.9, 'Site velocity: '+ str_bL + '$\pm$' + str_b95CI+' mm/year', ha='center', va='center', transform=fig1.transAxes,backgroundcolor='1',alpha=1)
       fig1.text(0.1, 0.07, '$SE_b$= '+ str_SEb + ' mm/year', ha='center', va='center', transform=fig1.transAxes)
       fig1.text(0.3, 0.07, '$SE_{bc}$= '+ str_SEbc + ' mm/year', ha='center', va='center', transform=fig1.transAxes)
       fig1.text(0.7, 0.07, 'Calculated vs. Propjected 95%CI: '+ str_b95CI + ' vs. '+ str_b95CI_mod + ' mm/year', ha='center', va='center', transform=fig1.transAxes)
   

       fig2.plot(year, Ri,'.',c='0.5')
       fig2.plot(year, NLi, 'r.')
       fig2.text(0.1, 0.07, '$b_{NL}$= '+ str_bNL + ' mm/year', ha='center', va='center', transform=fig2.transAxes)
       fig2.text(0.1, 0.9, 'RMS: '+ str_RMS_rm_L + ' mm', ha='center', va='center', transform=fig2.transAxes)

       fig3.plot(xt, y_con,'.',c='0.5')
       fig3.plot(xt, Si,'r.')
       fig3.set_ylim(bottom=min(y_con)*1.2, top=max(y_con)*1.2)
       fig3.text(0.1, 0.07, '$b_S$= '+ str_bS + ' mm/year', ha='center', va='center', transform=fig3.transAxes)
       fig3.text(0.6,0.07, '$S_i$='+str_c0+str_c1+'cos(2$\pi$($t_i$-$t_0$))' + str_d1+'sin(2$\pi$($t_i$-$t_0$))'+str_c2+'cos(4$\pi$($t_i$-$t_0$))'+str_d2+'sin(4$\pi$($t_i$-$t_0$))', ha='center', va='center', transform=fig3.transAxes)
       fig3.text(0.1, 0.9, 'RMS: '+ str_RMS_rm_LNL + ' mm', ha='center', va='center', transform=fig3.transAxes)
       fig3.text(0.8, 0.9, 'Peak-to-Trough Amplitude: '+ str_P2T + ' mm', ha='center', va='center', transform=fig3.transAxes)
       
       fig4.plot(xt, ri,'r.')
       fig4.text(0.1, 0.9, 'RMS: '+ str_RMS_ri + ' mm', ha='center', va='center', transform=fig4.transAxes)

       fig1.set_ylabel('Dis. (cm)')
       fig2.set_ylabel('Dis. (cm)')
       fig3.set_ylabel('Dis. (cm)')
       fig4.set_ylabel('Dis. (cm)')

       fig4.set_xlabel('Year', labelpad=12, fontsize=15)

       fig1.set_title('(a) Displacements y(i) and the Linear Component L(i)')
       fig2.set_title('(b) Non-Linear Component NL(i)')
       fig3.set_title('(c) Seasonal Component S(i)')
       fig4.set_title('(d) Residuals r(i)')

       fig.savefig(GNSS + '_' + DIR + '_Decomposition.png')
       fig.savefig(GNSS + '_' + DIR + '_Decomposition.pdf')
  
       
       # output the time series, original and filled
       f1_out = GNSS + "_" + DIR + "_Linear_NonLinear.txt"
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
       f2_out = GNSS + "_" + DIR + "_SeasonalM.txt"
       df = pd.concat([xt, y_con*10,Si*10,ri*10], axis=1)
       df.columns = ['Year_con', 'Dis_filled', 'SeasonMod','Final_Res']
       df.to_csv(f2_out, header=True, index=None, sep=' ', mode='w', float_format='%.5f')
       
       # output all parameters
       f3_out = GNSS + "_" + DIR + "_AllParameters.txt"
      
       Columns=['GNSS','ENU','Vel(mm/y)','95%CI(mm/y)','Projected_95%CI(mm/y)','N','Neff', \
                'b_NL(mm/y)', 'b_S(mm/y)', 'SEb(mm/y)', 'SEbc(mm/y)','RMS_rm_L(mm)','RMS_rm_LNL(mm)','RMS_rm_LNLS(mm)', \
                'c0(cm)','c1(cm)','d1(cm)','c2(cm)','d2(cm)','Annual_Amp(mm)','Semi_Annual_Amp(mm)', 'P2T_Amp (mm)']
       Parameters=[b_L*10, b95CI*10,b95CI_mod*10, N, Neff, \
                   b_NL*10,b_S*10,SEb*10,SEbc*10,RMS_rm_L,RMS_rm_LNL,RMS_ri, \
                   c0,c1,d1,c2,d2,P1*10,P2*10,P2T*10]
       with open(f3_out, 'w') as f:
            for col in Columns:
                f.write(col)
                f.write(',')
       with open(f3_out, 'a') as f: 
            f.write('\n') 
            f.write(GNSS)
            f.write(',')
            f.write(DIR)
            f.write(',')
            for val in Parameters:
                f.write(str(round(val,2)))
                f.write(',')
      
    else: 
       print('Output is off!')  
       
    if pltshow == 'on':
       plt.show()

    return b_L, b95CI
 







    
  




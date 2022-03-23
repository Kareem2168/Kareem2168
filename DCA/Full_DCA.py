import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def hyperbolic(t, qi, di, b):

  return qi / (np.abs((1 + b * di * t))**(1/b))

def arps_fit_hyp(t, q, plot=None):
  
  
  import datetime
  from scipy.optimize import curve_fit
  

  def hyperbolic(t, qi, di, b):
    return qi / (np.abs((1 + b * di * t))**(1/b))
  

  
  date = t
  timedelta = [j-i for i, j in zip(t[:-1], t[1:])]
  timedelta = np.array(timedelta)
  timedelta = timedelta / datetime.timedelta(days=1)

  
  t = np.cumsum(timedelta)
  t = np.append(0, t)
  t = t.astype(float)

  t_normalized = t / max(t)
  q_normalized = q / max(q)  

 
  popt, pcov = curve_fit(hyperbolic, t_normalized, q_normalized)
  qi, di, b = popt

  

  
  qi = qi * max(q)
  di = di / max(t)

  if plot==True:
    # Print all parameters and RMSE
    print('Initial production rate (qi)  : {:.5f} VOL/D'.format(qi))
    print('Initial decline rate (di)     : {:.5f} VOL/D'.format(di))
    print('Decline coefficient (b)       : {:.5f}'.format(b))
     

  

    
    tfit = np.linspace(min(t), max(t), 100)
    qfit = hyperbolic(tfit, qi, di, b)

    # Plot data and hyperbolic curve
    plt.figure(figsize=(10,7))

    plt.step(t, q, color='blue', label="Data")
    plt.plot(tfit, qfit, color='red', label="Hyperbolic Curve")
    plt.title('Decline Curve Analysis', size=20, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Rate (STB/d)')
    plt.xlim(min(t), max(t)); plt.ylim(ymin=0)

    plt.legend()
    plt.grid()
    plt.show()
    if 0<b<1:

      print('Hyperbolic Function')
    elif "b=0":
      print('Use the Arps_fit_har')
    elif "b=1":
      print('Use the Arps_fit_exp')


  return qi, di, b
def harmonic(t, qi, di):
  return qi / (np.abs(1+(di * t)))
def arps_fit_har(t, q, plot=None):
  
  
  import datetime
  from scipy.optimize import curve_fit
  

  def harmonic(t, qi, di):
    return qi / (np.abs(1+(di * t)))
  

  # subtract one datetime to another datetime
  date = t
  timedelta = [j-i for i, j in zip(t[:-1], t[1:])]
  timedelta = np.array(timedelta)
  timedelta = timedelta / datetime.timedelta(days=1)

  # take cumulative sum over timedeltas
  t = np.cumsum(timedelta)
  t = np.append(0, t)
  t = t.astype(float)

  
  t_normalized = t / max(t)
  q_normalized = q / max(q)  

  
  popt, pcov = curve_fit(harmonic, t_normalized, q_normalized)
  qi, di = popt

  

  # De-normalize qi and di
  qi = qi * max(q)
  di = di / max(t)

  if plot==True:
    
    print('Initial production rate (qi)  : {:.5f} VOL/D'.format(qi))
    print('Initial decline rate (di)     : {:.5f} VOL/D'.format(di))
    
    

  

    
    tfit = np.linspace(min(t), max(t), 100)
    qfit = harmonic(tfit, qi, di)

    
    plt.figure(figsize=(10,7))

    plt.step(t, q, color='blue', label="Data")
    plt.plot(tfit, qfit, color='red', label="Harmonic Curve")
    plt.title('Decline Curve Analysis', size=20, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Rate (STB/d)')
    plt.xlim(min(t), max(t)); plt.ylim(ymin=0)

    plt.legend()
    plt.grid()
    plt.show()
    

  return qi, di
def exponential(t, qi, di):
  return qi / (np.abs(qi * (np.exp(-di * t))))
def arps_fit_exp(t, q, plot=None):
  
  
  import datetime
  from scipy.optimize import curve_fit
  

  def exponential(t, qi, di):
    return qi / (np.abs(qi * (np.exp(-di * t))))
  

  # subtract one datetime to another datetime
  date = t
  timedelta = [j-i for i, j in zip(t[:-1], t[1:])]
  timedelta = np.array(timedelta)
  timedelta = timedelta / datetime.timedelta(days=1)

  # take cumulative sum over timedeltas
  t = np.cumsum(timedelta)
  t = np.append(0, t)
  t = t.astype(float)

  # normalize the time and rate data
  t_normalized = t / max(t)
  q_normalized = q / max(q)  

  popt, pcov = curve_fit(exponential, t_normalized, q_normalized)
  qi, di = popt

  qi = qi * max(q)
  di = di / max(t)

  if plot==True:
    # Print all parameters and RMSE
    print('Initial production rate (qi)  : {:.5f} VOL/D'.format(qi))
    print('Initial decline rate (di)     : {:.5f} VOL/D'.format(di))
   
      

  

    
    tfit = np.linspace(min(t), max(t), 100)
    qfit = exponential(tfit, qi, di)

   
    plt.figure(figsize=(10,7))

    plt.step(t, q, color='blue', label="Data")
    plt.plot(tfit, qfit, color='red', label="Exponential Curve")
    plt.title('Decline Curve Analysis', size=20, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Rate (STB/d)')
    plt.xlim(min(t), max(t)); plt.ylim(ymin=0)

    plt.legend()
    plt.grid()
    plt.show()
    

  return qi, di

def remove_outlier(df, column_name, window, number_of_stdevs_away_from_mean, trim=False):
 

  df[column_name+'_rol_Av']=df[column_name].rolling(window=window, center=True).mean()
  df[column_name+'_rol_Std']=df[column_name].rolling(window=window, center=True).std()

  
  df[column_name+'_is_Outlier']=(abs(df[column_name]-df[
                              column_name+'_rol_Av'])>(
                              number_of_stdevs_away_from_mean*df[
                              column_name+'_rol_Std']))
  

  result = df.drop(df[df[column_name+'_is_Outlier'] == True].index).reset_index(drop=True)

  
  result = result[result[column_name+'_rol_Av'].notna()]  

  if trim==True:
    
    maxi = result[column_name+'_rol_Av'].max()
    maxi_index = (result[result[column_name+'_rol_Av']==maxi].index.values)[0]
    result = result.iloc[maxi_index:,:].reset_index(drop=True)

  return result  


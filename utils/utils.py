#basic libraries
import pandas as pd
import numpy as np

#statistics libraries
import scipy
from  scipy.stats import  boxcox
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA


#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
import plotly.figure_factory as ff 
from plotly.offline import iplot
from plotly import tools
    
    
def plot_line(data, y, title='', range_x=None):
    fig = px.line(data, y=y, range_x=range_x)
    fig['layout'] = dict(title=dict(text=title,  font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18)),
                         yaxis=dict(title='Value', titlefont=dict(size=18)))
    fig.show()
    
    
def plot_hist(data, title='', range_x=None):
    fig = px.histogram(data, range_x=range_x)
    fig['layout'] = dict(title=dict(text=title.title(), font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18)),
                         yaxis=dict(title='Count', titlefont=dict(size=18)))
    fig.show()

    
def test_stationarity(df, series, title='', ret_values=None):
    
    # Determing rolling statistics
    rolmean = df[series].rolling(window = 12, center = False).mean().dropna() #Checkif our data has constant mean
    rolstd = df[series].rolling(window = 12, center = False).std().dropna()   #Checkif our data has constant variance
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('===>Results of Dickey-Fuller Test for %s:\n' %(series))
    dftest = sts.adfuller(df[series].dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    print(f'Result: The series is {"not " if dfoutput[1] > 0.05 else ""}stationary')
    
    # Plot rolling statistics:
    fig = go.Figure()
    fig.add_scatter(x=tuple(range(len(df[series]))), y=df[series].values, name='TS Data')
    fig.add_scatter(x=tuple(range(len(rolmean.values))), y=rolmean.values, name='Rolling Mean')
    fig.add_scatter(x=tuple(range(len(rolstd.values))), y=rolstd.values, name='Rolling std')
    fig['layout'] = dict(title=title.title(), titlefont=dict(size=20),
                         xaxis=dict(title='Range', titlefont=dict(size=18)),
                         yaxis=dict(title='Values', titlefont=dict(size=18)))
    fig.show()
    
    if ret_values:
        return dfoutput[1]
    
def create_corr_plot(series, series_name='', plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05, method='ols', nlags=40) if plot_pacf\
    else acf(series.dropna(), alpha=0.05, fft=False, nlags=40) #nlags=10*np.log10(len(series))
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][1:][x]), mode='lines', line_color='#3f3f3f', 
                     name='lag{}'.format(x)) for x in range(len(corr_array[0][1:]))]
    
    [fig.add_scatter(x=[i], y=[corr_array[0][1:][i]], mode='markers', marker_color='#1f77b4', 
                     marker_size=12,name='lag{}'.format(i)) for i in np.arange(len(corr_array[0][1:]))]
    
    fig.add_scatter(x=np.arange(len(corr_array[0][1:])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)',
                    name='Confidence interval')
    
    fig.add_scatter(x=np.arange(len(corr_array[0][1:])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                    fill='tonexty', line_color='rgba(255,255,255,0)', name='Confidence interval')
    
    fig.add_hrect(y0=2/((len(series)**0.5)), y1=-2/((len(series)**0.5)), line_width=0, fillcolor="red", opacity=0.5)

    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,42])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation function (PACF) of {}'.format(series_name) if plot_pacf\
     else 'Autocorrelation function (ACF) of {}'.format(series_name)
    
    fig['layout']=dict(title=title,titlefont=dict(size=20), width=1050)
    
    fig.show()

<a id="outlines"></a>

------------------------------------------
------------------------------------------
<h2 align="center" style='color:#0077EE'>Time Series Forecasting using Auto ARIMA & Deep Learning Algorithms</h2><br> 

------------------------------------------
------------------------------------------

In this repository, I'm going to explain how to do time series forecasting and analysis in Python. Also, I'll implement some deep learning techniques used for forecasting using the Keras framework.


<h1><span>&#8226;</span> Outline:</h1>
<ul>
    <li><a href="#gather_data">Gathering the data</a>
    <li><a href="#EDA">Look at the data</a>  
    <li><a href="#stationarity">Data stationarity analysis</a>
        <ul>
            <li><a href="#test">Test for Stationarity</a>
            <li><a href="#data_trans">Apply some data transformation for Stationarity</a>   
        </ul>
    <li><a href="#decomposition ">TS Decomposition</a>
    <li><a href="#corr">See the Correlation between Our Time Series.</a>
    <li><a href="#acf_pacf">Plot the ACF & PACF For the data.</a>
    <li><a href="#forecasting">Forecasting</a>
        <ul>
            <li><a href="#auto_arima">Auto ARIMA model</a>
            <li><a href="#DL">Deep Learning Algorithms</a>
                <ul>
                    <li><a href="#mlp">MLP approaches</a>
                        <ul>
                            <li><a href="#uni">Univariate Forecasting</a>
                            <li><a href="#multi">Multivariate Forecasting</a>
                                <ul>
                                    <li><a href="#multi_in">Multiple Input</a>
                                        <ul>
                                        <li><a href="#single_dense">Single Dense</a>
                                        <li><a href="#multi_headed">Multi-headed</a>
                                        </ul>
                                    <li><a href="#multiple_parallel">Multiple Parallel</a>
                                        <ul>
                                        <li><a href="#vector_output">Vector-Output</a>
                                        <li><a href="#multi_output">Multi-Output</a>
                                        </ul>
                                </ul>
                            <li><a href="#multi_step">Multi-Step Forcasting</a>
                                <ul>
                                    <li><a href="#multiple_in_multi_step">Multiple Input</a>
                                    <li><a href="#multiple_parallel_multi_step">Multiple Parallel</a>
                                </ul>
                        </ul>
                </ul>
                <li><a>CNN</a>
        <ul>
            <li><a href="#uni">Univariate Forecasting</a>
            <li><a href="#multi">Multivariate Forecasting</a>
                <ul>
                    <li><a href="#multiple_in">Multiple Input</a>
                        <ul>
                        <li><a href="#single cnn">Single CNN</a>
                        <li><a href="#multi_headed">Multi-headed</a>
                        </ul>
                    <li><a href="#multiple_parallel">Multiple Parallel</a>
                        <ul>
                        <li><a href="#vector_output">Vector-Output</a>
                        <li><a href="#multi_output">Multi-Output</a>
                        </ul>
                </ul>
            <li><a href="#multi_step">Multi-Step Forcasting</a>
                <ul>
                    <li><a href="#uni_mult_step">Univariate Multi-Step></a>
                    <li><a href="#multi_mult_step">Multivariate Multi-Step</a>
                        <ul>
                            <li><a href="#mult_step_multi_in">Multiple Input</a>
                            <li><a href="#mult_step_multiple_parallel">Multiple Parallel</a>
                        </ul>
        </ul>

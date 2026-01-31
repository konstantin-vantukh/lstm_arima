The updates cover LSTM engine only, the ARIMA part must stay untouched. Write a separate **./plans/architecture-delta.md** for review, do not modify any code at this stage, I will modify it later. The **architecture-delta.md** document must be the input for the Orchestrator. Do not call the Orchestrator, I will call it later. Do not modify the existing **./plans/archtecture.md** document.

The changes in LSTM engine must be the following:

The create_rolling_windows() procedure in ./src/lstm_engine.py must split the input data into windows 350 ticks long with a 175 ticks stride, so that a 50% overlap between windows exists.
The length in ticks and stride must be defined in the configuration file, these are the default values.

The LSTM network must have the following architecture: 
* LSTM layer of 20 nodes and return_sequences set to True, 
* Hidden LSTM layer of 10 nodes and return_sequences set to **False**, 
* Dropout layer with dropout rate of 0.2. The dropout rate must be defined in the configuration file, the value given here is a default. 
* Dense layer with number of units equal to the value of **--horizon** key. 

Make changes to the ./forecaster.py. Replace the recursive forecasting with direct usage of the output of the Dense layer. It has as many units as the forecasting horizon is.
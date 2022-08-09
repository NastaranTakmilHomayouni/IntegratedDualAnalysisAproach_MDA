import mne
import numpy as np                 # for multi-dimensional containers
import pandas as pd                # for DataFrames
from plotly import graph_objects as go  # for data visualisation
from plotly import io as pio            # to set shahin plot layout
import os
pio.templates['shahin'] = pio.to_templated(go.Figure().update_layout(yaxis=dict(autorange = "reversed"),margin=dict(t=0,r=0,b=40,l=40))).layout.template
pio.templates.default = 'shahin'
pio.renderers.default = "notebook_connected"
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
app = Flask(__name__)
@app.route('/')
def load_data():
    epochs = mne.io.read_epochs_eeglab('sampleEEGdata.mat').crop(-0.2,1)
    values = epochs.to_data_frame()
    values = values.groupby("time").mean()
    values.head()
    fig = go.Figure(layout=dict(xaxis=dict(title='time'), yaxis=dict(title='voltage')))
    for ch in epochs.info['ch_names']:
        fig.add_scatter(x=epochs.times, y=values[ch], name=ch)
    #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.show()


if __name__ == '__main__':
    app.run(debug=True)
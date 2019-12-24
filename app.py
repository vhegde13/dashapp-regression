# -*- coding: utf-8 -*-

#Created on Thu Dec 19 23:55:18 2019
#App displays at http://127.0.0.1:8050/
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly_express as px
import sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from dash.dependencies import Input, Output

bostonArr = datasets.load_boston()
boston = pd.DataFrame(data=bostonArr['data'],columns=bostonArr['feature_names'])
boston['MEDV'] = bostonArr['target']


col_options = [dict(label=x,value=x) for x in boston.columns]

feature_options = [dict(label=x,value=x) for x in boston.drop('MEDV',axis=1).columns]

dimensions = ["x"]

features = ["f"]



app = dash.Dash(__name__)

server = app.server
	
app.layout = html.Div(
    [dcc.Tabs(id="tabs-example", value="tab1-example",children=[ 
        
        dcc.Tab(label="EDA",value="tab1-example",
                children=[html.H1("Dash App Tutorial EDA of BostonHousing using Plotly"),
        html.Div(
            [
                html.P(["Compare Median House Value with other features"
                        + ":", dcc.Dropdown(id=d, options=col_options)])
                for d in dimensions
            ],
            style={"width": "25%", "float": "left"},
        ),
        dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"})]),
        dcc.Tab(label="Prediction",value="tab2-example",
                children=[html.H1("Linear Prediction of Boston Housing"),
                          html.Div(
                            [html.P(["Choose Features"
                                     +":",dcc.Dropdown(id="fslct",
                                         options=feature_options,multi=True)])
                             for fslct in features]),
                            
                            html.Div(
                            html.P(["Select Train-Test Split"
                                     +":",dcc.Slider(id="slider",
                                         min=0,
                                         max=1,
                                         step=0.05,
                                         marks={i / 10: str(i / 10) for i in range(0, 10)},
                                         value=0.8)] ),
                            
                              
                              ),
                            html.Table([
        html.Tr([html.Th(['Training RMSE']), html.Td(id='rmsetrn')]),
        html.Tr([html.Th(['Training R', html.Sup(2)]), html.Td(id='rtrn')]),
        html.Tr([html.Th(['Test RMSE']), html.Td(id='rmsetst')]),
        html.Tr([html.Th(['Test R', html.Sup(2)]), html.Td(id='rtst')]),
    ])])])])
    




@app.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
def make_figure(x):
    fig = px.scatter(
        boston,
        x=x,
        y="MEDV",
        trendline="ols",
        marginal_y="box",
        marginal_x="box",
        height=700,
    )
    fig.update_xaxes(matches=None)
    return fig

@app.callback([Output('rmsetrn','children'),
              Output('rtrn','children'),
              Output('rmsetst','children'),
              Output('rtst','children')],
              [Input("slider",'value'),
               Input("fslct",'value')])
def show_rmse(split,features):
    bostonN = boston[features]
    X=bostonN
    Y=boston['MEDV']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split, random_state=5)
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    # model evaluation for training set
    y_train_predict = lin_model.predict(X_train)
    rmse_trn = round((np.sqrt(mean_squared_error(Y_train, y_train_predict))),2)
    r2_trn = round(r2_score(Y_train, y_train_predict),2)
    
    # model evaluation for testing set
    y_test_predict = lin_model.predict(X_test)
    rmse_tst = round((np.sqrt(mean_squared_error(Y_test, y_test_predict))),2)
    r2_tst = round(r2_score(Y_test, y_test_predict),2)
    
    return rmse_trn,r2_trn,rmse_tst,r2_tst



if __name__ == '__main__':
    app.run_server(debug=False)
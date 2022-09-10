# -*- coding: utf-8 -*-

# We start with the import of standard ML librairies
import pandas as pd
import numpy as np
import math

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output


# We start by creating a virtual regression use-case
X, y = make_regression(n_samples=1000, n_features=8, n_informative=5, random_state=22)

# We rename columns as industrial parameters
col_names = ["Temperature","Viscosity","Pressure", "pH","Inlet_flow", "Rotating_Speed","Particles_size","Color_density"]

df = pd.DataFrame(X, columns=col_names)

# We change the most important features ranges to make them look like actual figures
df["pH"]=6.5+df["pH"]/4
df["Pressure"]=10+df["Pressure"]
df["Temperature"]=20+df["Temperature"]
df["Y"] = 90+y/20

# We train a simple RF model
model = RandomForestRegressor()
model.fit(df.drop("Y", axis=1), df["Y"])

# We create a DataFrame to store the features' importance and their corresponding label
df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=col_names)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)
# The command below can be activated in a standard notebook to display the chart
#fig_features_importance.show()

# We record the name, min, mean and max of the three most important features
slider_1_label = df_feature_importances.index[0]
slider_1_min = math.floor(df[slider_1_label].min())
slider_1_mean = round(df[slider_1_label].mean())
slider_1_max = round(df[slider_1_label].max())

slider_2_label = df_feature_importances.index[1]
slider_2_min = math.floor(df[slider_2_label].min())
slider_2_mean = round(df[slider_2_label].mean())
slider_2_max = round(df[slider_2_label].max())

slider_3_label = df_feature_importances.index[2]
slider_3_min = math.floor(df[slider_3_label].min())
slider_3_mean = round(df[slider_3_label].mean())
slider_3_max = round(df[slider_3_label].max())
app = dash.Dash()

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output

# We apply basic HTML formatting to the layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        # Title display
                        html.H1(children="Simulation Tool"),
                        
                        # Dash Graph Component calls the fig_features_importance parameters
                        dcc.Graph(figure=fig_features_importance),
                        
                        # We display the most important feature's name
                        html.H4(children=slider_1_label),

                        # The Dash Slider is built according to Feature #1 ranges
                        dcc.Slider(
                            id='X1_slider',
                            min=slider_1_min,
                            max=slider_1_max,
                            step=0.5,
                            value=slider_1_mean,
                            marks={i: '{} bars'.format(i) for i in range(slider_1_min, slider_1_max+1)}
                            ),

                        # The same logic is applied to the following names / sliders
                        html.H4(children=slider_2_label),

                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=0.5,
                            value=slider_2_mean,
                            marks={i: '{}°'.format(i) for i in range(slider_2_min, slider_2_max+1)}
                        ),

                        html.H4(children=slider_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=0.1,
                            value=slider_3_mean,
                            marks={i: '{}'.format(i) for i in np.linspace(slider_3_min, slider_3_max,1+(slider_3_max-slider_3_min)*5)},
                        ),
                        
                        # The predictin result will be displayed and updated here
                        html.H2(id="prediction_result"),

                    ])
                    # The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([X1,
                       df["Viscosity"].mean(),
                       df["Particles_size"].mean(),
                       X2,
                       df["Inlet_flow"].mean(),
                       df["Rotating_Speed"].mean(),
                       X3,
                       df["Color_density"].mean()]).reshape(1,-1)        
    
    # Prediction is calculated based on the input_X array
    prediction = model.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Prediction: {}".format(round(prediction,1))

if __name__ == "__main__":
    app.run_server()
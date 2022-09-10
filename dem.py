import pandas as pd
import csv
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc

file1=genfromtxt("sym.csv",delimiter=',',dtype='str')


dic={}
count=0
for val in file1:
    if val[0] not in dic:
        dic[val[0]]=count
        count+=1
#print("mm",dic)        
for val in file1:
    val[0]=dic[val[0]]
#print(file1) 

trainingSet=file1
#testingSet=file2

trainingX=trainingSet[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
trainingX=trainingX.astype(int)
trainingY=trainingSet[:,[0]]

#testingX=testingSet[:,[1,2,3,4,5,6,7,8,9,10,11]]
    #testingX=testingX.astype(float)

lr=linear_model.LogisticRegression()
lr.fit(trainingX,trainingY.ravel())
'''l=[0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1]
a=int(lr.predict([l]))
print(a)
for x in dic:
    if(dic[x]==a):
        print("you might be suffering from %s"%x)'''

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
                
                )
app.config['suppress_callback_exceptions'] = False                
app.layout = dbc.Container([

dbc.Row(
        dbc.Col(html.H1("Dementia Dashboard",
                        className='text-center text-primary mb-4 align-self-start',style={'borderRadius':'22px','font_family':  "Courier New",'borderBottomStyle':"solid","borderBottomWidth":"2px","borderBottomColor":'gray',"height":"80px","marginTop":"30px"}),
                width=12)
    ),


dbc.Row([dbc.Col([
           html.H1("Symptoms Checker"),
  dbc.Form(
                    [
                        dbc.FormGroup(
                            [
                               
                dbc.Checklist(id="cl",
    options=[
        {'label': 'Sleep Disorder','value':1},
        {'label': 'Mood Disturbance','value':2},
        {'label': 'Neuropsychiatric', 'value': 3},
        {'label': 'Reduced driving abilities ', 'value': 4},
        {'label': 'Behavioral and psychologic ', 'value': 5},
        {'label': 'Agitation', 'value': 6},
        {'label': 'Apathy', 'value': 7},
        {'label': 'eating problems', 'value': 8},
        {'label': 'feeding', 'value': 9},
        {'label': 'Urinary Continence', 'value': 10},
        {'label': 'Delusion', 'value': 11},
        {'label': 'Agitation', 'value': 12},
        {'label': 'Depression', 'value': 13},
        {'label': 'Hallucination', 'value': 14},
        {'label': 'Wandering', 'value': 15},
        {'label': 'Anxiety', 'value': 16}



    ],
    #value=[0,1],
    style={"padding":"24px"}
    )
                        
    ]
                        )
                        
                        
                        
                        
                        
                        
                        ],className="ont-weight-normal",style={'backgroundColor':'#F8F9F9','padding':'10px'})


]),


]),
 dbc.Row(
        dbc.Col(html.H3("",id="Res",
                        className='text-center mb-4 card-text  text-success border border-primary',style={'borderRadius':'22px','font_family':  "Courier New",'padding':'22px'}),
                width=12)
                
    )
])

@app.callback(Output(component_id="Res",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
            
             [Input(component_id='cl', component_property='value')]
              
              
              
              
              
              
              
              )

# The input variable are set in the same order as the callback Inputs
def update_prediction(x1):
    m=list(x1)
    l=[]
    for i in range(1,17):
        l.insert(i,0)
    for k in m:
        l[k]=1
    #l=list(x1)
    print(l)
    a=int(lr.predict([l]))
    print(a)
    for x in dic:
        if(dic[x]==a):
            print("you might be suffering from %s"%x)
            return(x)
    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
   






if __name__=='__main__':
    app.run_server(debug=True, port=8000)




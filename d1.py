#sore_throat,fever,swelling_of_body,dizziness,headache,bodyache,rash,fatigue,chills,muscleache,coughing
import pandas
import csv
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output


#file2=genfromtxt("fill",delimiter=',',dtype='int')
file1=genfromtxt("ds.csv",delimiter=',',dtype='str')

dic={}
count=0
for val in file1:
    if val[11] not in dic:
        dic[val[11]]=count
        count+=1
#print(dic)        
for val in file1:
    val[11]=dic[val[11]]
  
trainingSet=file1
#testingSet=file2

trainingX=trainingSet[:,[0,1,2,3,4,5,6,7,8,9,10]]
trainingX=trainingX.astype(float)
trainingY=trainingSet[:,[11]]

#testingX=testingSet[:,[1,2,3,4,5,6,7,8,9,10,11]]
    #testingX=testingX.astype(float)

lr=linear_model.LogisticRegression()
lr.fit(trainingX,trainingY)
#l=[1,1,0,0,0,0,0,0,0,1,1]
#a=int(lr.predict([l]))
#print(a)
'''for x in dic:
    if(dic[x]==a):
        print("you might be suffering from %s"%x)'''
app = dash.Dash()        
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        # Title display
                        html.H1(children="dementia"),
                        html.Div(["Input: ",
              dcc.Input(id='my-input1',type='number',placeholder="Bp rate 1 to 5"),
              dcc.Input(id='my-input2', value=0,type='text'),
              dcc.Input(id='my-input3', value=0,type='text'),
              dcc.Input(id='my-input4', value=0,type='text'),
              dcc.Input(id='my-input5', value=0,type='text'),
              dcc.Input(id='my-input6', value=0,type='text'),
              dcc.Input(id='my-input7', value=0,type='text'),
              dcc.Input(id='my-input8', value=0,type='text'),
              dcc.Input(id='my-input9', value=0,type='text'),
              dcc.Input(id='my-input10', value=0,type='text'),
              dcc.Input(id='my-input11', value=0,type='text')
              
              
              
              
              ]),
    
                    
                        html.H2(id="prediction_result"),

                    ])
                    # The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("my-input1","value"), Input("my-input2","value"), Input("my-input3","value"),
              Input("my-input4","value"),Input("my-input5","value"),Input("my-input6","value"),Input("my-input7","value"),
              Input("my-input8","value"),
              Input("my-input9","value"),Input("my-input10","value"),Input("my-input11","value")
              
              
              
              
              
              
              ])

# The input variable are set in the same order as the callback Inputs
def update_prediction(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    print(x1)
    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    L=[int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8),int(x9),int(x10),int(x11)]
    a=int(lr.predict([L]))
    print(a)
    for x in dic:
        if(dic[x]==a):
        #print("you might be suffering from %s"%x)
    
    # And retuned to the Output of the callback function
            return "Prediction: {}".format(x)

if __name__ == "__main__":
    app.run_server()
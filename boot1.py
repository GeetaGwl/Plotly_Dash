import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn import preprocessing
from sklearn import utils
from numpy import genfromtxt
from sklearn import linear_model
sns.set()
import pandas as pd
import csv
import numpy as np

df = pd.read_csv('./oasis_longitudinal.csv')
print(df.head())
Gen={'M': 0,'F': 1} 
df.Gen= [Gen[item] for item in df.Gen] 

df = df.fillna(0)
print(df)

Y = df['CDR'].values # Target for the model
X = df[['Gen','Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(Y))
print(utils.multiclass.type_of_target(Y.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))
lr=linear_model.LogisticRegression()
lr.fit(X,training_scores_encoded)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
                
                )
app.config['suppress_callback_exceptions'] = False                
app.layout = dbc.Container([

dbc.Row(
        dbc.Col(html.H1("Dementia Dashboard",
                        className='text-center text-primary mb-4 align-self-start',style={'borderRadius':'22px','font_family':  "Courier New"}),
                width=12)
    ),

    dbc.Row([


        dbc.Col([
           
  dbc.Form(
                    [
                        dbc.FormGroup(
                            [
                               dbc.Label("Gender:"),
                dbc.RadioItems(
                    id="Gen1",
                    options=[
                      
                        {"label": "Male", "value": 0},
                        {"label": "Female", "value": 1},
                    ],
                    inline=True,
                    value=True,
                )]
                        ),
                       dbc.Row([dbc.Col([
                        dbc.FormGroup(
                            [
                                dbc.Label("Age", className="mr-2"),
                                dbc.Input(type="number", placeholder="Enter Age",id="Age1",value=0)
                               
                            ],
                            className="mr-3"
                       )],width=6),dbc.Col([
                         dbc.FormGroup(
                            [
                                dbc.Label("Education", className="mr-2"),
                                dbc.Input(id="edu1",type="number", placeholder="Education in years",value=0),
                            ],
                            className="mr-3")],width=6
                       )]

                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("SES", className="mr-2"),
                               dcc.Slider(
id="ses1",
min=1,
max=5,
value=1,
step=1,
 marks={i: str(i) for i in range(1,6)})
,
                            ],
                            className="mr-3",
                        ),
                        dbc.Row([dbc.Col([

 dbc.FormGroup(
                            [
                                dbc.Label("MMSE", className="mr-2"),
                                dbc.Input(id="mmse1",type="number", placeholder="From 0 to 30",min=0,max=30,value=0)
 ,dbc.FormText(
            "(range is from 0 = worst to 30 = best)",
            color="secondary",
        )
                               

                            ],
                            className="mr-3",
                        )],width=6),dbc.Col([
dbc.FormGroup(
                            [
                                dbc.Label("eTIV", className="mr-2"),
                                dbc.Input(id="etiv1",type="number",value=0)
                               
,dbc.FormText(
            "(eg. 1987)",
            color="secondary",
        )
                            ],
                            className="mr-3",
                        )],width=6)]),
                        dbc.Row([dbc.Col([
dbc.FormGroup(
                            [
                                dbc.Label("nWBV", className="mr-2"),
                                dbc.Input(id="nwbv1",type="number",value=0)

                               
,dbc.FormText(
            "(eg. 0.696)",
            color="secondary",
        )
                            ],
                            className="mr-3",
                        )],width=6),dbc.Col([
dbc.FormGroup(
                            [
                                dbc.Label("ASF", className="mr-2"),
                                dbc.Input(id="asf1",type="number",value=0)

                               
,dbc.FormText(
            "(eg. 0.876)",
            color="secondary",
        )
                            ],
                            className="mr-3",
                        )],width=6)]),







                      
                    ],className="ont-weight-normal",style={'backgroundColor':'#F8F9F9','padding':'10px'}
                    
                ),

               dbc.Row(
        dbc.Col(html.H3("",id="Res",
                        className='text-center mb-4 card-text  text-success border border-primary',style={'borderRadius':'22px','font_family':  "Courier New",'padding':'22px'}),
                width=12)
                
    )
    
        ], #width={'size':5, 'offset':0, 'order':2},
           width=5
        ),   
        dbc.Col([
            dbc.Card(
                [
                    dbc.CardBody(
dcc.Markdown('''
**Attributes:
*It consists of 15 attributes which are describes as follows :



**M.F** - Gende  

**Hand** - Handedness  

**Age** - Age in years  

**EDUC** - Years of education  

**SES** - Socioeconomic status as assessed by the Hollingshead Index of Social Position and classified into categories from 1 (highest status) to 5 (lowest status)

**Clinical Info**

**MMSE** - Mini-Mental State Examination score (range is from 0 = worst to 30 = best)  

 **eTIV** - Estimated total intracranial volume, mm3  

**nWBV** - Normalized whole-brain volume, expressed as a percent of all voxels in the atlas-masked image that are labeled as gray or white matter by the automated tissue segmentation process 

**ASF** - Atlas scaling factor (unitless). Computed scaling factor that transforms native-space brain and skull to the atlas target (i.e., the determinant of the transform matrix)  
'''),



                            
                            className="card-text bg-info text-white")
                    
                  
                       
                ],
                style={"width": "40rem"},
            )
        ], #width={'size':5, 'offset':1},
           width=7
        )
    ]),  # Vertical: start, center, end
    

])
             
@app.callback(Output(component_id="Res",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("Gen1","value"), Input("Age1","value"), Input("edu1","value"),
              Input("ses1","value"),Input("mmse1","value"),Input("etiv1","value"),Input("nwbv1","value"),
              Input("asf1","value")
              
              
              
              
              
              
              
              ])

# The input variable are set in the same order as the callback Inputs
def update_prediction(x1,x2,x3,x4,x5,x6,x7,x8):
    print(x1)
    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    L=[int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8)]
    b=int(lr.predict([L]))
    if(b==0):
        x="Normal"
        print("Normal \n No Need")
        print("No Need")
    elif(b==1):
        x="Mild"
        print("Mild")

    elif(b==2):
        x="Moderate"
        print("Moderate")
    else:
        x="severe "
        print("severe")   
    return "Stage: {}".format(x)

if __name__=='__main__':
    app.run_server(debug=True, port=8000)

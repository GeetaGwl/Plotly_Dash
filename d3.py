import pandas as pd
import csv
from sklearn import linear_model
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn import preprocessing
from sklearn import utils
import seaborn as sns
import dash_bootstrap_components as dbc
sns.set()

df = pd.read_csv('./oasis_longitudinal.csv')
print(df.head())
Gen={'M': 0,'F': 1} 
df.Gen= [Gen[item] for item in df.Gen] 

df = df.fillna(0)
#print(df)

Y = df['CDR'].values # Target for the model
X = df[['Gen','Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)

lr=linear_model.LogisticRegression()
lr.fit(X,training_scores_encoded)
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="primary",
    dark=True,
)
app = dash.Dash()  
app.layout=html.Div(children=[html.H1(children="Dementia Dashboard",style={'color':'darkblue','textAlign': 'center'}),
html.Div(children=[html.Label(children="Gender :  ",style={'float':'left','marginLeft':'50px','fontSize':'30px'}),dcc.Dropdown(id="Gen",options=[{'label':'Male','value':0},{'label':'Female','value':1}],style={'width':'200px','float':'left'}),
html.Br(),
html.Label(children="Age :  ",style={'float':'left','marginLeft':'50px','fontSize':'30px'}),dcc.Input(id="Age",style={'width':'200px','float':'left'}),
html.Br(),
html.Label(children="Education :  ",style={'float':'left','marginLeft':'50px','fontSize':'30px'}),dcc.Input(id="Educ",style={'width':'200px','float':'left'})



],style={'width':'60%'})])
                       


if __name__ == "__main__":
    app.run_server(debug=True)

import dash
import dash_html_components as html
import dash_core_components as dcc
app = dash.Dash()
app.layout=html.Div([
html.Label("select item"),
html.Br(),
html.Hr(),
dcc.Dropdown(
id="D1",
options=[
    {'label':'USA','value':'USA'},
    {'label':'India','value':'Ind'},
    {'label':'China','value':'Ch'},
    {'label':'Aust','value':'A'},
    {'label':'India','value':'Ind'}

    
],
placeholder="select one",


),


dcc.Slider(
min=1,
max=10,
value=5,
step=0.5,
 marks={i: str(i) for i in range(1,11)}




),
 html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),
html.Label('Text Input'),
    dcc.Input(value='MTL', type='text',style={'color':'red'}),





])



if __name__ == '__main__':
    app.run_server(debug = True)
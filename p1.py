import dash
import dash_html_components as html
import dash_core_components as dcc

#Code to start an application
app = dash.Dash()

#HTML layout and Graph components are defined in this section
app.layout = html.Div([html.H1(children='Sample Dash Web App Dashboard',style={
    'backgroundColor':'red','color':'blue'
}),html.H2("hello world"),
html.Div(
dcc.Graph(id='dash_graph_2',
figure = {'data': [{'x':[1,2,3,4,5], 'y':[4,6,3,8,1], 'type': 'bar', 'name':'Aeroplane'},
{'x':[1,2,3,4,5], 'y':[9,3,1,9,4], 'type': 'bar', 'name':'Car'},
{'x':[1,2,3,4,5], 'y':[4,6,3,8,1], 'type': 'line', 'name':'Train'},],
'layout':{'plot_bgcolor':'yellow','title': 'Dash Example App 2'}
} 
),style={'width':'80%'}),html.P(children="its...",style={'display':'inline'})
                                ])



                                


#Code to run the application
if __name__ == '__main__':
    app.run_server(debug = True)
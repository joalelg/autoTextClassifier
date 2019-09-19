## Adding one more component, no style yet

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table_experiments as dt

app = dash.Dash()
# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501
df = pd.read_csv('../data/display_df.csv')
print(df.tail(1).T)

#  Layouts
layout_table = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
)
layout_table['font-size'] = '12'
layout_table['margin-top'] = '20'

app.layout = html.Div(
     html.Div([
        html.Div([
            html.H1(children='Text Analyzer', className = 'eight columns'),

                html.Img(
                    src="https://www.fadoq.ca/wp-content/uploads/2017/06/bdebcoul-1024x212.jpg",
                    className='four columns',
                    style={
                        'height': '16%',
                        'width': '16%',
                        'float': 'right',
                        'position': 'relative',
                        'padding-top': 12,
                        'padding-right': 0
                    },
                ),
                 ], className = 'row'
                 )
     , html.Div([
            html.Div(id='text-content',
                children='''
                        Dash: A web application framework for Python.
                    ''', className = 'ten columns'),
                ], className = 'row'
              )

     , html.Div([ 
         dcc.Graph(
            id='ReuceDimPlot',
            figure={

                'data': [
                    # Hover creating a dic for each category using list comprehension
              		{
              			'x': df.loc[df.Category==c,'Dim1'], 
                    	'y': df.loc[df.Category==c,'Dim2'],
                    	'Predicted': df.loc[df.Category==c,'Prediction'],
                    	'Athletics': df.loc[df.Category==c,'athletics'],
                    	'Cricket': df.loc[df.Category==c,'cricket'],
                    	'Football': df.loc[df.Category==c,'football'],
                    	'Rugby': df.loc[df.Category==c,'rugby'],
                    	'Tennis': df.loc[df.Category==c,'tennis'],    
                    	'mode': 'markers', 
                    	'name': c}
                        for c in sorted(df.Category.unique()) ],
                'layout': {
                    'xaxis': {'title': 'Dim1'},
                    'yaxis': {'title': "Dim2"}
                            }

                        }
                    , className = 'eight columns' 
                   ),
                html.Div(
                    [
                        dt.DataTable(
                            rows=df.to_dict('records'),
                            columns=df.columns,
                            row_selectable=True,
                            filterable=True,
                            sortable=True,
                            selected_row_indices=[],
                            id='datatable'),
                    ],
                    style = layout_table,
                    className="four columns"
                )
                  ], className = 'row')
    , html.Div(
               [
        html.Div(id='text-content-long',
            children='''
                    
                ''', className = 'twelve columns'
                 )
                 ], className = 'row') 
    , html.Div([
                html.Div(
                    [
                        html.P('Developed by  José Alejandro - ', style = {'display': 'inline'}),
                        html.A('josalelg@hotmail.com', href = 'mailto:josalelv@hotmail.com')
                    ], className = "twelve columns",
                       style = {'fontSize': 18, 'padding-top': 20}
                )
            ], className="row")
             ], className='ten columns offset-by-one')

)



@app.callback(
    dash.dependencies.Output('text-content', 'children'),
    [dash.dependencies.Input('ReuceDimPlot', 'hoverData')])
def update_text(hoverData):
    print(type(hoverData))  #Hovver attributes dic including columns 'x', 'y', etc as keys
    s = df[df['Dim1'] == hoverData['points'][0]['x']]
    return html.H3(s['Prediction'])


@app.callback(
    dash.dependencies.Output('text-content-long', 'children'),
    [dash.dependencies.Input('ReuceDimPlot', 'hoverData')])
def update_text(hoverData):
    print(type(hoverData))
    s = df[df['Dim1'] == hoverData['points'][0]['x']]
    return html.P( 'Result {}'.format(list(s['Text'])[0]))

#@app.callback(
#    dash.dependencies.Output('datatable', 'rows'),
#    [dash.dependencies.Input('ReduceDimPlot', 'relayoutData')])
#def update_table(relayoutData):
#    print('printa ',relayoutData)
#    #s = df[df['Dim1'] == relayoutData['points'][0]['x']]
#    #rows = map_s.to_dict('records')
#    #return rows

@app.callback(
    dash.dependencies.Output('datatable', 'rows'),
    [dash.dependencies.Input('ReuceDimPlot', 'relayoutData')])
def update_table(selectedData):
    
    if selectedData:
        #s = selectedData['range']['x'] #Min and max from 'selectedData' instead of relayoutData
        sx = (selectedData['xaxis.range[0]'], selectedData['xaxis.range[1]'])
        sy = (selectedData['yaxis.range[0]'], selectedData['yaxis.range[1]'])
        sx_mask = [x for x in df['Dim1'] if (x >= sx[0] and x <= sx[1])]
        sy_mask = [y for y in df['Dim2'] if (y >= sy[0] and y <= sy[1])]
        s = df.loc[df['Dim1'].isin(sx_mask) & df['Dim2'].isin(sy_mask),:] 
        rows = s.to_dict('records')
    else:
        rows = df.copy().to_dict('records')
    return rows


if __name__ == '__main__':
    app.run_server(debug=True)

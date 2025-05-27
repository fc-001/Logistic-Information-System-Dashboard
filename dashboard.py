#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dash import Dash, html, dcc, callback, Output, Input, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hub2hub import fig_n2rdc, fig_rdc2rdc, fig_rdc2store
from operations_metrics import fig_op1, fig_op2, fig_op3, ship_all
from inventory import result, agg as risk_agg, fig_risk, fig_sankey

df_area = (
    ship_all
    .groupby(['month','region'], as_index=False)['qty']
    .sum()
    .pivot(index='month', columns='region', values='qty')
    .fillna(0)
    .sort_index()
)
fig_perf = make_subplots(
    rows=2, cols=1,
    row_heights=[0.6, 0.4],
    specs=[[{"type":"xy"}], [{"type":"domain"}]],
    subplot_titles=("月度区域发运量", "区域累计发运量占比")
)
for reg in df_area.columns:
    fig_perf.add_trace(
        go.Scatter(x=df_area.index, y=df_area[reg],
                   mode='lines', stackgroup='one', name=reg),
        row=1, col=1
    )
totals = df_area.sum().to_dict()
fig_perf.add_trace(
    go.Pie(labels=list(totals.keys()), values=list(totals.values()),
           textinfo='label+percent'),
    row=2, col=1
)
fig_perf.update_layout(margin={'l':40,'r':20,'t':50,'b':20})

stores = pd.read_csv('data/stores.csv', encoding='utf-8')
stores_meta = stores[['store_name','region']].rename(columns={'store_name':'dest'})
tot2sub   = pd.read_csv('data/ship_tot2sub.csv')[['source','dest','month','qty']]
sub2sub   = pd.read_csv('data/ship_sub2sub.csv')[['source','dest','month','qty']]
sub2store = pd.read_csv('data/ship_sub2store.csv')[['source','dest','month','qty']] \
    .merge(stores_meta, on='dest', how='left')
tot2sub['region']  = None
sub2sub['region']  = None
ship_all = pd.concat([tot2sub, sub2sub, sub2store], ignore_index=True)
ship_all['route']  = ship_all['source'] + '→' + ship_all['dest']
all_regions        = ['全部'] + sorted(ship_all['region'].dropna().unique().tolist())
all_routes         = ['全部'] + sorted(ship_all['route'].unique().tolist())






app = Dash(__name__)

app.layout = html.Div(style={
    'display': 'grid',
    'height': '100vh',
    'gridTemplateColumns': '1fr 2fr 1fr',
    'gridTemplateRows':    '2fr 3fr 5fr',
    'gridTemplateAreas': (
        '"inventory  maps   sankey"'
        ' "monthly    maps   sankey"'
        ' "risk       metrics2 perf"'
    ),
    'gap': '8px'
}, children=[

    # 左上：库存预警
    html.Div(style={'gridArea':'inventory','padding':10,'overflow':'auto'}, children=[
        html.H4("库存情况", style={'marginTop':0}),
        dash_table.DataTable(
            id='inventory-table',
            columns=[
                {'name':'分仓','id':'node'},
                {'name':'当前库存','id':'actual_inv','type':'numeric','format':{'specifier':',.0f'}},
                {'name':'安全库存 SS','id':'SS','type':'numeric','format':{'specifier':',.0f'}},
                {'name':'订货点 R','id':'R','type':'numeric','format':{'specifier':',.0f'}},
                {'name':'最优批量 Q*','id':'EOQ','type':'numeric','format':{'specifier':',.0f'}},
            ],
            data=result.to_dict('records'),
            page_size=10,
            style_cell={'textAlign':'center','padding':'5px'},
            style_header={'backgroundColor':'#f9f9f9','fontWeight':'bold'},
            style_data_conditional=[
                {'if':{'filter_query':'{actual_inv} < {SS}','column_id':'actual_inv'},
                 'backgroundColor':'#FFCCCC','color':'#900'},
                {'if':{'filter_query':'{actual_inv} >= {SS} && {actual_inv} < {R}','column_id':'actual_inv'},
                 'backgroundColor':'#FFE5CC','color':'#A60'},
                {'if':{'filter_query':'{actual_inv} = 0','column_id':'actual_inv'},
                 'backgroundColor':'#FF0000','color':'white','fontWeight':'bold'},
            ],
            style_table={'height':'100%'}
        )
    ]),

    # 左中：发运量月变化
    html.Div(style={'gridArea':'monthly','padding':10,'overflow':'auto'}, children=[
        html.H4("发运量月变化", style={'marginTop':0}),
        html.Div([
            html.Div([
                html.Label("按区域筛选："),
                dcc.Dropdown(id='region-filter',
                             options=[{'label':r,'value':r} for r in all_regions],
                             value='全部')
            ], style={'width':'48%','display':'inline-block'}),
            html.Div([
                html.Label("按路线筛选："),
                dcc.Dropdown(id='route-filter',
                             options=[{'label':rt,'value':rt} for rt in all_routes],
                             value='全部')
            ], style={'width':'48%','display':'inline-block','marginLeft':'4%'})
        ]),
        dcc.Graph(id='monthly-chart', style={'height':'100%','marginTop':'10px'})
    ]),

    # 左下：风险预警看板
    html.Div(style={'gridArea':'risk','padding':10,'overflow':'auto'}, children=[
        html.H4("风险预警看板", style={'marginTop':0}),
        dash_table.DataTable(
            id='risk-table',
            columns=[
                {'name':'路线','id':'route'},
                {'name':'总发运量','id':'total_qty','type':'numeric','format':{'specifier':','}},
                {'name':'延误次数','id':'delay_cnt','type':'numeric'},
                {'name':'延误率','id':'delay_rate','type':'numeric','format':{'specifier':'.1%'}},
                {'name':'亏损次数','id':'cost_overrun_cnt','type':'numeric'},
                {'name':'平均成本','id':'avg_cost','type':'numeric','format':{'specifier':',.0f'}},
            ],
            data=risk_agg.to_dict('records'),
            page_size=10,
            style_cell={'textAlign':'center','padding':'5px'},
            style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'},
            style_data_conditional=[
                {'if':{'filter_query':'{delay_rate} > 0.1'}, 'backgroundColor':'#FFE5E5'},
                {'if':{'filter_query':'{delay_rate} <= 0.1 && {cost_overrun_cnt} > 0'},
                 'backgroundColor':'#FFF5E5'},
            ],
            style_table={'height':'40%'}
        ),
        dcc.Graph(figure=fig_risk, style={'height':'55%','marginTop':'8px'})
    ]),

    # 中侧：线路地图
    html.Div(style={'gridArea':'maps','padding':10,'overflow':'auto'}, children=[
        html.H4("线路地图", style={'marginTop':0}),
        dcc.RadioItems(id='map-selector',
                       options=[
                           {'label':'NDC → RDC','value':'n2r'},
                           {'label':'RDC → RDC','value':'r2r'},
                           {'label':'RDC → Store','value':'r2s'},
                       ],
                       value='n2r', inline=True),
        dcc.Graph(id='map-graph', style={'height':'100%','marginTop':'10px'})
    ]),

    # 右侧：补货漏斗图
    html.Div(style={'gridArea':'sankey','padding':10,'overflow':'auto'}, children=[
        html.H4("羽绒服流向", style={'marginTop':0}),
        dcc.Graph(figure=fig_sankey, style={'height':'100%','marginTop':'10px'})
    ]),

    # 中下：发运量趋势
    html.Div(style={'gridArea':'metrics2','padding':10,'overflow':'auto'}, children=[
        html.H4("月度指标", style={'marginTop':0}),
        dcc.RadioItems(id='op-metrics-selector',
                       options=[
                           {'label':'发运量变化','value':'op1'},
                           {'label':'成本和服务水平变化','value':'op3'},
                       ],
                       value='op1', inline=True),
        dcc.Graph(id='op-metrics-graph', style={'height':'90%','marginTop':'10px'})
    ]),

    # 右下：绩效概览
    html.Div(style={'gridArea':'perf','padding':10,'overflow':'auto'}, children=[
        html.H4("绩效概览", style={'marginTop':0}),
        dcc.Graph(figure=fig_perf, style={'height':'100%','marginTop':'10px'})
    ]),
])


@app.callback(
    Output('monthly-chart', 'figure'),
    Input('region-filter', 'value'),
    Input('route-filter', 'value'),
)
def update_main_chart(selected_region, selected_route):
    df = ship_all.copy()
    if selected_region != '全部':
        df = df[df['region'] == selected_region]
    if selected_route != '全部':
        df = df[df['route'] == selected_route]
    df_month = df.groupby('month', as_index=False)['qty'].sum()
    if df_month.empty:
        fig = px.line(title="无符合条件的数据", template='plotly_white')
        fig.update_layout(xaxis={'visible':False}, yaxis={'visible':False})
        return fig

    fig = px.bar(
        df_month,
        x='month', y='qty',
        labels={'qty':'发运量','month':'月份'},
        title="按月发运量走势",
        template='plotly_white',
        color_discrete_sequence=['#5470C6'],
    )
    fig.add_scatter(
        x=df_month['month'], y=df_month['qty'],
        mode='lines+markers',
        name='trend',
        line=dict(color='#EE6666', width=3, shape='spline'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate="月份 %{x}<br>发运量 %{y}<extra></extra>"
    )
    fig.update_layout(
        font=dict(family='"Helvetica Neue", Arial, sans-serif', size=14, color='#333'),
        xaxis=dict(tickmode='linear', tickangle=-45),
        yaxis=dict(title="发运量", tickformat=','),
        bargap=0.25,
        margin={'l':60,'r':20,'t':80,'b':80},
        plot_bgcolor='white'
    )
    return fig

@app.callback(
    Output('map-graph','figure'),
    Input('map-selector','value')
)
def swap_map(mode):
    return {'n2r':fig_n2rdc,'r2r':fig_rdc2rdc,'r2s':fig_rdc2store}[mode]

@app.callback(
    Output('op-metrics-graph','figure'),
    Input('op-metrics-selector','value')
)
def swap_op(mode):
    return fig_op1 if mode=='op1' else fig_op3

if __name__ == '__main__':
    app.run(debug=True, port=8001)
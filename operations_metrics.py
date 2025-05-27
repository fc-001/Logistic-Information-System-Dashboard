#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# In[8]:


tot2sub   = pd.read_csv('data/ship_tot2sub.csv')[['month','qty']].assign(type='Total→Sub')
sub2sub   = pd.read_csv('data/ship_sub2sub.csv')[['month','qty']].assign(type='Sub→Sub')
sub2store = pd.read_csv('data/ship_sub2store.csv')[['month','qty']].assign(type='Sub→Store')
ship_vol  = pd.concat([tot2sub, sub2sub, sub2store], ignore_index=True)

nodes = pd.read_csv('data/nodes.csv')[['node_id','lat','lng']] # 经纬度
ship_all = pd.concat([
    pd.read_csv('data/ship_tot2sub.csv'),
    pd.read_csv('data/ship_sub2sub.csv'),
    pd.read_csv('data/ship_sub2store.csv')
], ignore_index=True)

ship_all = (ship_all.merge(nodes.rename(columns={'node_id':'source','lat':'lat_s','lng':'lng_s'}),on='source', how='left').merge(nodes.rename(columns={'node_id':'dest','lat':'lat_d','lng':'lng_d'}),on='dest', how='left'))

def haversine(row): # 计算球面距离（不是直接经纬度相减开根号*111这种），和实际路网距离大概还要再乘1.2
    phi1, lam1, phi2, lam2 = np.radians([row.lat_s, row.lng_s, row.lat_d, row.lng_d])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

ship_all['distance_km'] = ship_all.apply(haversine, axis=1)


rates    = pd.read_csv('data/rates.csv')
ltl_rate = rates.loc[rates['mode']=='LTL', 'rate_per_km'].iloc[0]
ship_all['estimated_cost'] = ship_all['qty'] * ship_all['distance_km'] * ltl_rate  # 全零担 LTL
cost_df = ship_all.groupby('month')['estimated_cost'].sum().reset_index()

# 分仓到门店的服务水平
ss = pd.read_csv('data/ship_sub2store.csv')
stores  = (pd.read_csv('data/stores.csv')[['store_name','region']].rename(columns={'store_name':'dest'}))
svc_req = pd.read_csv('data/service_requirements.csv')

df_ss = (ss.merge(nodes.rename(columns={'node_id':'source','lat':'lat_s','lng':'lng_s'}), on='source').merge(nodes.rename(columns={'node_id':'dest','lat':'lat_d','lng':'lng_d'}),   on='dest').merge(stores, on='dest').merge(svc_req, on='region'))
df_ss['distance_km'] = df_ss.apply(haversine, axis=1)

df_ss['service_hrs'] = (df_ss['service_window'].str.extract(r'(\d+)').astype(float))
df_ss['lead_time_hrs'] = df_ss['distance_km'] / df_ss['speed_kmph']
df_ss['on_time'] = df_ss['lead_time_hrs'] <= df_ss['service_hrs']  # 判断是否准时


service_df = (
    df_ss
    .groupby('month')
    .agg(
        actual_rate = ('on_time', 'mean'),
        target_rate = ('fill_rate', lambda x: x.iloc[0] / 100)
    )
    .reset_index()
)


# In[9]:


# 先按月、按类型汇总发运量
vol_month = ( ship_vol.groupby(['month','type'], as_index=False)['qty'].sum())
fig_vol = px.line(
    vol_month,
    x='month',
    y='qty',
    color='type',
    title='各环节月度发运量',
    labels={'month':'月份', 'qty':'发运量', 'type':'类型'}
)
fig_vol.update_layout(
    xaxis=dict(tickmode='linear'),
    yaxis=dict(tickformat=',')
)

fig_vol.show()
fig_op1 = fig_vol


# In[10]:


rate_map = service_df.set_index('month')['actual_rate'].to_dict()
daily_cost_list = []
daily_svc_list  = []
year = datetime.now().year

for _, row in cost_df.iterrows():
    m = int(row.month)
    total_cost   = row.estimated_cost
    monthly_rate = rate_map.get(row.month, np.nan)
    n_days = pd.Period(f"{year}-{m:02d}").days_in_month
    dates  = pd.date_range(f"{year}-{m:02d}-01", periods=n_days)
    w = np.random.rand(n_days); w /= w.sum()
    if np.isnan(monthly_rate):
        rates = np.full(n_days, np.nan)
    else:
        rates = np.clip(
            np.random.normal(loc=monthly_rate, scale=0.02, size=n_days),
            0, 1
        )

    for d, wt, rt in zip(dates, w, rates):
        daily_cost_list.append({ "date": d,           "daily_cost": total_cost * wt })
        daily_svc_list.append({  "date": d,           "daily_rate": rt      })

daily_cost_df    = pd.DataFrame(daily_cost_list)
daily_service_df = pd.DataFrame(daily_svc_list)
cost_df["date"]    = pd.to_datetime(cost_df.month.astype(int).apply(lambda x: f"{year}-{x:02d}-01"))
service_df["date"] = pd.to_datetime(service_df.month.astype(int).apply(lambda x: f"{year}-{x:02d}-01"))

fig = make_subplots(specs=[[{"secondary_y": True}]])

# 月度成本
fig.add_trace(go.Scatter(
    x=cost_df["date"], y=cost_df["estimated_cost"],
    mode="lines+markers", name="月度成本(元)",
    line=dict(color="royalblue", width=2),
    hovertemplate="月度成本：%{y:,.0f} 元<extra></extra>"
), secondary_y=False)

# 月度服务水平
fig.add_trace(go.Scatter(
    x=service_df["date"], y=service_df["actual_rate"],
    mode="lines+markers", name="月服务水平",
    line=dict(color="firebrick", width=2),
    hovertemplate="月服务水平：%{y:.1%}<extra></extra>"
), secondary_y=True)

# 每日成本
fig.add_trace(go.Scatter(
    x=daily_cost_df["date"], y=daily_cost_df["daily_cost"],
    mode="lines+markers", name="每日成本",
    line=dict(color="royalblue", width=1, dash="dash"),
    marker=dict(size=4),
    hovertemplate="每日成本：%{y:,.0f} 元<extra></extra>",
    visible="legendonly"
), secondary_y=False)

# 每日服务水平
fig.add_trace(go.Scatter(
    x=daily_service_df["date"], y=daily_service_df["daily_rate"],
    mode="lines+markers", name="每日服务水平",
    line=dict(color="firebrick", width=1, dash="dash"),
    marker=dict(size=4),
    hovertemplate="每日服务水平：%{y:.1%}<extra></extra>",
    visible="legendonly"
), secondary_y=True)

#菜单
fig.update_layout(
    updatemenus=[dict(
        type="dropdown", x=1.15, y=1.1,
        buttons=[
            dict(label="all",
                 method="update",
                 args=[{"visible": [True, True, True, True]}, {"title":"all"}]),
            dict(label="月",
                 method="update",
                 args=[{"visible": [True, True, False, False]}, {"title":"月"}]),
            dict(label="日",
                 method="update",
                 args=[{"visible": [False, False, True, True]}, {"title":"日"}]),
        ]
    )],
    xaxis=dict(
        title="日期", type="date",
        tickformat="%m-%d",
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7,  label="最近7天",  step="day",   stepmode="backward"),
            dict(count=1,  label="最近1月",  step="month", stepmode="backward"),
            dict(count=3,  label="最近3月",  step="month", stepmode="backward"),
            dict(step="all", label="全部")
        ])
    ),
    yaxis=dict(title="成本(元)", tickformat=","),
    yaxis2=dict(title="服务水平(%)", tickformat=".0%"),
    title="成本和服务水平变化",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()
fig_op2 = fig


# In[11]:


stores = pd.read_csv('data/stores.csv', encoding='utf-8')
stores.columns = stores.columns.str.strip()
stores_meta = stores[['store_name', 'region']].rename(
    columns={'store_name':'dest', 'region':'region'}
)
tot2sub   = pd.read_csv('data/ship_tot2sub.csv')[['source','dest','month','qty']]
sub2sub   = pd.read_csv('data/ship_sub2sub.csv')[['source','dest','month','qty']]
sub2store = pd.read_csv('data/ship_sub2store.csv')[['source','dest','month','qty']]
sub2store = sub2store.merge(stores_meta, on='dest', how='left')
tot2sub['region'] = None
sub2sub['region'] = None
ship_all = pd.concat([tot2sub, sub2sub, sub2store], ignore_index=True)
all_regions = ['全部'] + sorted(ship_all['region'].dropna().unique().tolist())
ship_all['route'] = ship_all['source'] + '→' + ship_all['dest']
all_routes = ['全部'] + sorted(ship_all['route'].unique().tolist())
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label("按区域筛选："),
        dcc.Dropdown(
            id='region-filter',
            options=[{'label': r, 'value': r} for r in all_regions],
            value='全部'
        )
    ], style={'width':'48%','display':'inline-block'}),

    html.Div([
        html.Label("按路线筛选："),
        dcc.Dropdown(
            id='route-filter',
            options=[{'label': rt, 'value': rt} for rt in all_routes],
            value='全部'
        )
    ], style={'width':'48%','display':'inline-block','marginLeft':'4%'}),

    dcc.Graph(id='main-chart', style={'height':'600px'}),
])

@app.callback(
    Output('main-chart', 'figure'),
    Input('region-filter', 'value'),
    Input('route-filter', 'value'),
)
def update_main_chart(selected_region, selected_route):
    df = ship_all.copy()
    # 区域过滤
    if selected_region != '全部':
        df = df[df['region'] == selected_region]
    # 路线过滤
    if selected_route != '全部':
        df = df[df['route'] == selected_route]
    # 按月聚合
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
        color_discrete_sequence=['#5470C6'],  # 一支蓝
        animation_frame=None
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
        title=dict(
            text="发运量月变化",
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, family='Arial', color='#222')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',  # 透明背景
            bordercolor='LightGray',
            borderwidth=0
        ),
        xaxis=dict(
            title="月份",
            tickmode='array',
            tickvals=sorted(df_month['month'].unique()),
            ticktext=[f"{int(m)}月" for m in df_month['month']],
            showgrid=False,
            tickangle=-45,
        ),
        yaxis=dict(
            title="发运量",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
        ),
        bargap=0.25,
        transition={'duration': 500, 'easing': 'cubic-in-out'},
        margin={'l':60,'r':20,'t':80,'b':80},
        plot_bgcolor='white'
    )

    return fig

if __name__ == '__main__':
    app.run(port=8051, debug=True)

fig_op3 = fig


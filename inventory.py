#!/usr/bin/env python
# coding: utf-8

# In[11]:


import plotly.express as px
import dash
from dash import dcc, html, dash_table, Dash
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[20]:


def haversine(row):
    lat1, lon1, lat2, lon2 = map(np.radians, [row.lat_s, row.lng_s, row.lat_d, row.lng_d])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# 1.1 发运记录：子仓→门店
ship_ss    = pd.read_csv('data/ship_sub2store.csv')
# 节点经纬度
nodes      = pd.read_csv('data/nodes.csv')[['node_id','lat','lng']]
# 门店区域
stores     = pd.read_csv('data/stores.csv')[['store_name','region']]
stores_meta= stores.rename(columns={'store_name':'dest'})
# 服务要求
svc_req    = pd.read_csv('data/service_requirements.csv')[['region','speed_kmph','service_window','fill_rate']]
# 费率
rates      = pd.read_csv('data/rates.csv')[['mode','rate_per_km']]

# 1.2 合并、计算延误 & 超支标志
df = (
    ship_ss
    .merge(nodes.rename(columns={'node_id':'source','lat':'lat_s','lng':'lng_s'}), on='source', how='left')
    .merge(nodes.rename(columns={'node_id':'dest',  'lat':'lat_d','lng':'lng_d'}),   on='dest',   how='left')
    .merge(stores_meta, on='dest', how='left')
    .merge(svc_req,     on='region', how='left')
)
df['distance_km']       = 1.2*df.apply(haversine, axis=1)
df['service_window_hrs']= df['service_window'].str.extract(r'(\d+)').astype(float)
df['lead_time_hrs']     = df['distance_km'] / df['speed_kmph']
df['delay_flag']        = df['lead_time_hrs'] > df['service_window_hrs']

ltl_rate = rates.loc[rates['mode']=='LTL','rate_per_km'].iat[0]
df['estimated_cost']    = df['qty'] * df['distance_km'] * ltl_rate
threshold = df['estimated_cost'].mean() + df['estimated_cost'].std()
df['cost_overrun_flag'] = df['estimated_cost'] > threshold

df['route'] = df['source'] + '→' + df['dest']

# 1.3 聚合到路线层面
agg = df.groupby('route', as_index=False).agg(
    total_qty        = ('qty','sum'),
    delay_cnt        = ('delay_flag','sum'),
    delay_rate       = ('delay_flag','mean'),
    cost_overrun_cnt = ('cost_overrun_flag','sum'),
    avg_cost         = ('estimated_cost','mean')
)

# —— 2. 构造静态 fig_risk ——
# 2.1 构造延误最多 & 超支最多的两个子图
def build_fig_risk(dataframe):
    # 延误 Top5
    df1 = dataframe.sort_values('delay_cnt', ascending=False).head(5)
    fig1 = px.bar(df1, x='route', y='delay_cnt', labels={'delay_cnt':'延误次数','route':'路线'})

    # 超支 Top5
    df2 = dataframe.sort_values('cost_overrun_cnt', ascending=False).head(5)
    fig2 = px.bar(df2, x='route', y='cost_overrun_cnt', labels={'cost_overrun_cnt':'超支次数','route':'路线'})

    # 合并为上下两个子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("延误多发路线", "成本超支路线"),
        vertical_spacing=0.12
    )
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=False,
        margin={'l':40,'r':20,'t':50,'b':30},
    )
    return fig

fig_risk = build_fig_risk(agg)
app = Dash(__name__)

app.layout = html.Div([
    html.H2("风险预警看板", style={'textAlign':'center'}),
    dash_table.DataTable(
        id='risk-table',
        columns=[
            {'name':'路线','id':'route'},
            {'name':'发运量','id':'total_qty','type':'numeric'},
            {'name':'延误次数','id':'delay_cnt','type':'numeric'},
            {'name':'延误率','id':'delay_rate','type':'numeric','format':{'specifier':'.1%'}},
            {'name':'超支次数','id':'cost_overrun_cnt','type':'numeric'},
            {'name':'平均成本','id':'avg_cost','type':'numeric','format':{'specifier':',.0f'}}
        ],
        data=agg.to_dict('records'),
        page_size=10,
        style_cell={'textAlign':'center','padding':'5px'},
        style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'},
        style_data_conditional=[
            {'if': {'filter_query':'{delay_rate} > 0.1'},    'backgroundColor':'#FFE5E5'},
            {'if': {'filter_query':'{delay_rate} <= 0.1 && {cost_overrun_cnt} > 0'}, 'backgroundColor':'#FFF5E5'},
        ]
    ),
    html.Br(),
    dcc.Graph(figure=fig_risk, id='risk-chart')
])

@app.callback(
    Output('risk-chart','figure'),
    Input('risk-table','data')
)
def update_risk(rows):
    df_new = pd.DataFrame(rows)
    return build_fig_risk(df_new)

if __name__ == '__main__':
    app.run(port=8077, debug=True)


# In[13]:


"""
大前提是：1月初的库存为0
"""
policy = pd.read_csv('data/inventory_policy.csv')
h = policy.loc[0, 'holding_cost_pct'] / 100    # 年持有率
c = policy.loc[0, 'unit_value']                # 单位价值
K = policy.loc[0, 'fixed_cost']                # 每次订货固定成本
warehouses  = pd.read_csv('data/warehouses.csv')['warehouse_name'].tolist()

# 计算每个分仓的当期实际库存，入库量 = tot2sub + sub2sub
in1 = pd.read_csv('data/ship_tot2sub.csv')[['dest','qty']].rename(columns={'dest':'node','qty':'in'})
in2 = pd.read_csv('data/ship_sub2sub.csv')[['dest','qty']].rename(columns={'dest':'node','qty':'in'})
inbound = pd.concat([in1,in2],ignore_index=True).groupby('node',as_index=False)['in'].sum()
#
out1 = pd.read_csv('data/ship_sub2sub.csv')[['source','qty']].rename(columns={'source':'node','qty':'out'})
out2 = pd.read_csv('data/ship_sub2store.csv')[['source','qty']].rename(columns={'source':'node','qty':'out'})
outbound= pd.concat([out1,out2],ignore_index=True).groupby('node',as_index=False)['out'].sum()
inv_df = (pd.DataFrame({'node':warehouses[1:]}).merge(inbound,  on='node', how='left').merge(outbound, on='node', how='left').fillna(0))
inv_df['actual_inv'] = inv_df['in'] - inv_df['out']

# 阅读sub2store
ship_sub2sub   = pd.read_csv('data/ship_sub2sub.csv')[['source','month','qty']]
ship_sub2store = pd.read_csv('data/ship_sub2store.csv')[['source','month','qty']]
ship_out       = pd.concat([ship_sub2sub, ship_sub2store], ignore_index=True)
# 统计每个分仓每月出货总量
month_stats = (ship_out.groupby(['source','month'], as_index=False)['qty'].sum().rename(columns={'source':'node'}))
agg_month = (month_stats.groupby('node')['qty'].agg(mean_m='mean', std_m='std').reset_index())
agg_month['D_daily']     = agg_month['mean_m'] / 30
agg_month['sigma_daily'] = agg_month['std_m']  / 30


ship_t2s = pd.read_csv('data/ship_tot2sub.csv')[['source','dest','qty']]
t2s = (ship_t2s
    .merge(nodes.rename(columns={'node_id':'source','lat':'lat_s','lng':'lng_s'}), on='source')
    .merge(nodes.rename(columns={'node_id':'dest',  'lat':'lat_d','lng':'lng_d'}),   on='dest')
)
t2s['distance_km']    = t2s.apply(haversine, axis=1) # 距离
avg_speed_kmph = policy.loc[0,'throughput_max'] / 24  #这里直接用 policy 中的 throughput_max/24 来反推平均速度
t2s['lead_time_days'] = t2s['distance_km'] / avg_speed_kmph
L_days = t2s.groupby('dest', as_index=False)['lead_time_days'].mean().rename(columns={'dest':'node','lead_time_days':'L'}) # 平均提前起


Z98 = norm.ppf(0.98) # 服务水平98%

df = (agg_month
    .merge(L_days,     on='node', how='left')
    .merge(inv_df[['node','actual_inv']], on='node')
)
mean_L = df['L'].mean()
df = df.fillna({
    'D_daily':     0,
    'sigma_daily': 0,
    'L':           mean_L
})

df['SS'] = Z98 * df['sigma_daily'] * np.sqrt(df['L'])
df['R']  = df['D_daily']*df['L'] + df['SS']

# 年度需求
df['D_annual'] = df['mean_m'] * 12
# 年持有成本hc
hc = h * c
df['EOQ'] = np.sqrt(2 * df['D_annual'] * K / hc)
result = df[[
    'node','actual_inv','SS','R','EOQ'
]].round({
    'actual_inv':0,'SS':0,'R':0,'EOQ':0
})

app = Dash(__name__)

app.layout = html.Div([
    html.H2("库存情况"),
    dash_table.DataTable(
        id='inv-mgmt',
        columns=[
            {'name':'节点',         'id':'node'},
            {'name':'当前库存',     'id':'actual_inv', 'type':'numeric','format':{'specifier':',.0f'}},
            {'name':'安全库存 SS','id':'SS',         'type':'numeric','format':{'specifier':',.0f'}},
            {'name':'补货点 R',     'id':'R',          'type':'numeric','format':{'specifier':',.0f'}},
            {'name':'建议订货量 Q*','id':'EOQ',        'type':'numeric','format':{'specifier':',.0f'}},],
        data=result.to_dict('records'),
        style_cell={
            'textAlign':'center',
            'padding':'5px'
        },
        style_header={
            'backgroundColor':'#f9f9f9',
            'fontWeight':'bold'
        },
        style_data_conditional=[
            # 低于ss的标红
            {
                'if': {'filter_query':'{actual_inv} < {SS}', 'column_id':'actual_inv'},
                'backgroundColor':'#FFCCCC',
                'color':'#900'
            },
            # ss<inv<R 橙色
            {
                'if': {'filter_query':'{actual_inv} >= {SS} && {actual_inv} < {R}', 'column_id':'actual_inv'},
                'backgroundColor':'#FFE5CC',
                'color':'#A60'
            },
            # 库存为0，加醋标红
            {
                'if': {'filter_query':'{actual_inv} = 0', 'column_id':'actual_inv'},
                'backgroundColor':'#FF0000',
                'color':'white',
                'fontWeight':'bold'
            },
        ],
        page_size=10,
    )
])

if __name__ == '__main__':
    app.run(port=8055, debug=True)


# In[14]:


start_node = "NDC_Changshu"
nodes_list = result['node'].tolist()
stores = pd.read_csv('data/ship_sub2store.csv')['dest'].unique().tolist()
labels = [start_node] + nodes_list + stores
inbound = (
    pd.read_csv('data/ship_tot2sub.csv')[['dest','qty']]
    .rename(columns={'dest':'node','qty':'value'})
    .groupby('node', as_index=False).sum()
)
outbound = (
    pd.read_csv('data/ship_sub2store.csv')[['source','dest','qty']]
    .rename(columns={'source':'node','qty':'value'})
    .groupby(['node','dest'], as_index=False).sum()
)

source, target, value = [], [], []
for _, row in inbound.iterrows():
    if row['node'] in nodes_list:
        source.append(labels.index(start_node))
        target.append(labels.index(row['node']))
        value.append(row['value'])
for _, row in outbound.iterrows():
    if row['node'] in nodes_list and row['dest'] in stores:
        source.append(labels.index(row['node']))
        target.append(labels.index(row['dest']))
        value.append(row['value'])


n_dist = len(nodes_list)
n_store = len(stores)
node_x = [0.0] + [0.1]*n_dist + [0.8]*n_store
node_y = [0.5] \
         + list(np.linspace(0, 1, n_dist)) \
       + list(np.linspace(0, 1, n_store))

fig = go.Figure(go.Sankey(
    node=dict(
        label=labels,
        x=node_x,
        y=node_y,
        pad=15,
        thickness=20,
        color=["lightblue"] + ["lightgreen"]*n_dist + ["lightcoral"]*n_store
    ),
    link=dict(source=source, target=target, value=value)
))

fig.update_layout(
    title_text="总仓 → 分仓 → 门店羽绒服流向",
    font_size=10,
    width=1100, height=5000,
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.show()
fig_sankey = fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ship_st = pd.read_csv('data/ship_sub2store.csv')
nodes = pd.read_csv('data/nodes.csv')
# 2. 获取 RDC 列表和门店列表
rdcs   = pd.read_csv('data/warehouses.csv')['warehouse_name'].tolist()[1:]
stores = pd.read_csv('data/stores.csv')['store_name'].tolist()

# 3. 给节点打上类别标签
def label_node(n):
    if n in rdcs:
        return 'rdc'
    elif n in stores:
        return 'store'
    else:
        return 'other'

nodes['category'] = nodes['node_id'].apply(label_node)

# 4. 准备线路数据（只取分仓→门店）
lines = []
for _, row in ship_st.iterrows():
    src_id, dst_id, qty = row.source, row.dest, row.qty
    if src_id in rdcs and dst_id in stores:
        src = nodes.loc[nodes.node_id == src_id, ['lng','lat']].iloc[0]
        dst = nodes.loc[nodes.node_id == dst_id, ['lng','lat']].iloc[0]
        lines.append({
            'lon': [src.lng, dst.lng],
            'lat': [src.lat, dst.lat],
            'qty': qty
        })

# 5. 创建交互式图
fig = go.Figure()

# 5.1 绘制所有分仓→门店线路，只在图例显示一次
max_qty = max([l['qty'] for l in lines] + [1])
for i, l in enumerate(lines):
    fig.add_trace(go.Scattermap(
        lon = l['lon'],
        lat = l['lat'],
        mode = 'lines',
        line = dict(width = (l['qty']/max_qty)*10, color='green'),
        hovertemplate = f"Quantity: {l['qty']}<extra></extra>",
        name = 'RDC→Store',
        legendgroup = 'flows',
        showlegend = (i == 0)
    ))

# 5.2 绘制 RDC 节点
rdc_nodes = nodes[nodes.category=='rdc']
fig.add_trace(go.Scattermap(
    lon = rdc_nodes.lng,
    lat = rdc_nodes.lat,
    mode = 'markers',
    marker = dict(size=10, color='orange'),
    name = 'RDCs',
    hoverinfo = 'text',
    hovertext = rdc_nodes.node_id
))

# 5.3 绘制门店节点
store_nodes = nodes[nodes.category=='store']
fig.add_trace(go.Scattermap(
    lon = store_nodes.lng,
    lat = store_nodes.lat,
    mode = 'markers',
    marker = dict(size=8, color='blue'),
    name = 'Stores',
    hoverinfo = 'text',
    hovertext = store_nodes.node_id
))

# 6. 布局设置
fig.update_layout(
    mapbox_style  = "carto-positron",
    mapbox_center = dict(lat = nodes.lat.mean(), lon = nodes.lng.mean()),
    mapbox_zoom   = 4,
    margin        = dict(l=0, r=0, t=30, b=0),
    title         = "RDC → Store"
)

fig.show()
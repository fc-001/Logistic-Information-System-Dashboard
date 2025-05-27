#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go

# 载入节点和运输记录
nodes = pd.read_csv('data/nodes.csv')[['node_id','lat','lng']]
warehouses = pd.read_csv('data/warehouses.csv')['warehouse_name'].tolist()
origin = warehouses[0]
rdcs = warehouses[1:]
ship_tot2sub   = pd.read_csv('data/ship_tot2sub.csv')
ship_sub2sub   = pd.read_csv('data/ship_sub2sub.csv')
ship_sub2store = pd.read_csv('data/ship_sub2store.csv')

def make_n2r():
    fig = go.Figure()
    M = 1
    for _, row in ship_tot2sub[ship_tot2sub.source==origin].iterrows():
        if row.dest in rdcs:
            src = nodes.query("node_id==@origin").iloc[0]
            dst = nodes.query("node_id==@row.dest").iloc[0]
            M = max(M, row.qty)
            fig.add_trace(go.Scattermap(
                lon=[src.lng, dst.lng],
                lat=[src.lat, dst.lat],
                mode='lines',
                line=dict(width=(row.qty/M)*10, color='blue'),
                hovertemplate="Qty: %{customdata}<extra></extra>",
                customdata=[row.qty]
            ))
    # 添加节点标记
    fig.add_trace(go.Scattermap(
        lon=[nodes.query("node_id==@origin").lng.iloc[0]],
        lat=[nodes.query("node_id==@origin").lat.iloc[0]],
        mode='markers', marker=dict(size=14, color='red', symbol='diamond'),
        name='NDC'
    ))
    fig.add_trace(go.Scattermap(
        lon=nodes.query("node_id in @rdcs").lng,
        lat=nodes.query("node_id in @rdcs").lat,
        mode='markers', marker=dict(size=10, color='orange'),
        name='RDCs'
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center=dict(lat=nodes.lat.mean(), lon=nodes.lng.mean()),
        mapbox_zoom=4,
        margin=dict(l=0, r=0, t=30, b=0),
        title="NDC → RDC"
    )
    return fig

def make_r2r():
    fig = go.Figure()
    M = 1
    for _, row in ship_sub2sub.iterrows():
        if row.source in rdcs and row.dest in rdcs and row.source!=row.dest:
            src = nodes.query("node_id==@row.source").iloc[0]
            dst = nodes.query("node_id==@row.dest").iloc[0]
            M = max(M, row.qty)
            fig.add_trace(go.Scattermap(
                lon=[src.lng, dst.lng],
                lat=[src.lat, dst.lat],
                mode='lines',
                line=dict(width=(row.qty/M)*10, color='purple'),
                hovertemplate="Qty: %{customdata}<extra></extra>",
                customdata=[row.qty]
            ))
    fig.add_trace(go.Scattermap(
        lon=nodes.query("node_id in @rdcs").lng,
        lat=nodes.query("node_id in @rdcs").lat,
        mode='markers', marker=dict(size=10, color='orange'),
        name='RDCs'
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center=dict(lat=nodes.lat.mean(), lon=nodes.lng.mean()),
        mapbox_zoom=4,
        margin=dict(l=0, r=0, t=30, b=0),
        title="RDC → RDC"
    )
    return fig

def make_r2s():
    fig = go.Figure()
    M = 1
    stores = ship_sub2store.dest.unique().tolist()
    for _, row in ship_sub2store.iterrows():
        if row.source in rdcs and row.dest in stores:
            src = nodes.query("node_id==@row.source").iloc[0]
            dst = nodes.query("node_id==@row.dest").iloc[0]
            M = max(M, row.qty)
            fig.add_trace(go.Scattermap(
                lon=[src.lng, dst.lng],
                lat=[src.lat, dst.lat],
                mode='lines',
                line=dict(width=(row.qty/M)*10, color='green'),
                hovertemplate="Qty: %{customdata}<extra></extra>",
                customdata=[row.qty]
            ))
    fig.add_trace(go.Scattermap(
        lon=nodes.query("node_id in @rdcs").lng,
        lat=nodes.query("node_id in @rdcs").lat,
        mode='markers', marker=dict(size=10, color='orange'),
        name='RDCs'
    ))
    fig.add_trace(go.Scattermap(
        lon=nodes.query("node_id in @stores").lng,
        lat=nodes.query("node_id in @stores").lat,
        mode='markers', marker=dict(size=6, color='blue'),
        name='Stores'
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center=dict(lat=nodes.lat.mean(), lon=nodes.lng.mean()),
        mapbox_zoom=4,
        margin=dict(l=0, r=0, t=30, b=0),
        title="RDC → Store"
    )
    return fig

fig_n2rdc     = make_n2r()
fig_rdc2rdc  = make_r2r()
fig_rdc2store = make_r2s()
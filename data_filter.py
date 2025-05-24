import pandas as pd

# 加载 Excel
file_path = 'data.xlsx'
xls = pd.ExcelFile(file_path)

# 1. 仓库信息 (Sheet2)
df_warehouses = pd.read_excel(xls, sheet_name=1).rename(columns={
    '仓库名称':'warehouse_name',
    '仓库地址':'address',
    '仓库所在城市':'city',
    '仓库所在省份':'province',
    '仓库类型':'type',
    '纬度':'lat',
    '经度':'lng'
})
df_warehouses['warehouse_name'] = df_warehouses['warehouse_name'].str.strip()
df_warehouses = df_warehouses.dropna(subset=['warehouse_name','lat','lng']).drop_duplicates()
df_warehouses.to_csv('warehouses.csv', index=False)

# 2. 门店信息 (Sheet3)
df_stores = pd.read_excel(xls, sheet_name=2).rename(columns={
    '门店名称':'store_name',
    '门店类型':'type',
    '门店所在城市':'city',
    '门店所在省份':'province',
    '所在区域':'region',
    '纬度':'lat',
    '经度':'lng'
})
df_stores['store_name'] = df_stores['store_name'].str.strip()
df_stores = df_stores.dropna(subset=['store_name','lat','lng']).drop_duplicates()
df_stores.to_csv('stores.csv', index=False)

# 3. 发运记录 - 总仓→分仓 (Sheet4)
df_tot2sub = pd.read_excel(xls, sheet_name=3).rename(columns={
    'source':'source',
    'dest':'dest',
    '到达月份':'month_code',
    '数量':'qty'
})
df_tot2sub['source'] = df_tot2sub['source'].str.strip()
df_tot2sub['dest']   = df_tot2sub['dest'].str.strip()
df_tot2sub['qty']    = pd.to_numeric(df_tot2sub['qty'], errors='coerce').fillna(0)
df_tot2sub['month'] = (df_tot2sub['month_code']
    .astype(str)
    .str.extract(r'(\d{3})')[0]
    .astype(int))
df_tot2sub.to_csv('ship_tot2sub.csv', index=False)

# 4. 发运记录 - 分仓→分仓 (Sheet5)
df_sub2sub = pd.read_excel(xls, sheet_name=4).rename(columns={
    '调出分仓':'source',
    '调入分仓':'dest',
    '到达月份':'month_code',
    '数量之合计':'qty',
    '产品类别':'sku'
})
df_sub2sub['source'] = df_sub2sub['source'].str.strip()
df_sub2sub['dest']   = df_sub2sub['dest'].str.strip()
df_sub2sub['qty']    = pd.to_numeric(df_sub2sub['qty'], errors='coerce').fillna(0)
df_sub2sub['sku']    = df_sub2sub['sku'].astype(str).str.strip()
df_sub2sub['month']  = (df_sub2sub['month_code']
    .astype(str)
    .str.extract(r'(\d{3})')[0]
    .astype(int))
df_sub2sub.to_csv('ship_sub2sub.csv', index=False)

# 5. 发运记录 - 分仓→门店 (Sheet6)
df_sub2store = pd.read_excel(xls, sheet_name=5).rename(columns={
    '分仓名称':'source',
    '门店名称':'dest',
    '实际到货月份':'month_code',
    '数量':'qty',
    '产品类别':'sku'
})
df_sub2store['source'] = df_sub2store['source'].str.strip()
df_sub2store['dest']   = df_sub2store['dest'].str.strip()
df_sub2store['qty']    = pd.to_numeric(df_sub2store['qty'], errors='coerce').fillna(0)
df_sub2store['sku']    = df_sub2store['sku'].astype(str).str.strip()
df_sub2store['month']  = (df_sub2store['month_code']
    .astype(str)
    .str.extract(r'(\d{3})')[0]
    .astype(int))
df_sub2store.to_csv('ship_sub2store.csv', index=False)

# 6. 参考运价率 (Sheet7) — 手动构造
df_rates = pd.DataFrame([
    {'mode':'FTL','unit':'元/车·公里','rate_per_km':2.5},
    {'mode':'LTL','unit':'元/件·公里','rate_per_km':0.0055}
])
df_rates.to_csv('rates.csv', index=False)

# 7. 参考库存策略 (Sheet8)
df8 = pd.read_excel(xls, sheet_name=7, header=None)
df8[0] = df8[0].fillna('')
holding_cost_pct = float(df8.iloc[1,1])
throughput_min, throughput_max = map(int, df8.iloc[1,2].split('-'))
turns_low  = int(df8.iloc[1,3])
turns_high = int(df8.iloc[2,3])
unit_value = float(df8[df8[0].str.contains('产品价值')][1].iloc[0].replace('元',''))
fixed_cost = float(df8[df8[0].str.contains('分仓固定运营成本')][1].iloc[0].replace('元',''))
max_capacity = int(df8[df8[0].str.contains('分仓最大库容')][1].iloc[0].replace('件',''))
df_inv_params = pd.DataFrame([{
    'holding_cost_pct': holding_cost_pct,
    'throughput_min': throughput_min,
    'throughput_max': throughput_max,
    'turns_low': turns_low,
    'turns_high': turns_high,
    'unit_value': unit_value,
    'fixed_cost': fixed_cost,
    'max_capacity': max_capacity
}])
df_inv_params.to_csv('inventory_policy.csv', index=False)

# 8. 服务要求 (Sheet9)
df_service = pd.read_excel(xls, sheet_name=8).rename(columns={
    '区域':'region',
    '平均时速（KM/HR)':'speed_kmph',
    '服务时间':'service_window',
    '需求满足率':'fill_rate'
})
df_service['fill_rate'] = (df_service['fill_rate']
    .astype(str).str.rstrip('%').astype(float) / 100)
df_service.to_csv('service_requirements.csv', index=False)

# 9. 构建网络节点列表
nodes_ship = pd.DataFrame(pd.unique(
    pd.concat([
        df_tot2sub[['source','dest']],
        df_sub2sub[['source','dest']],
        df_sub2store[['source','dest']]
    ]).values.ravel()
), columns=['node_id'])
nodes_ware  = df_warehouses.rename(columns={'warehouse_name':'node_id'})[['node_id','lat','lng','type']]
nodes_store = df_stores.rename(columns={'store_name':'node_id'})[['node_id','lat','lng','type']]
nodes = (nodes_ship
    .merge(nodes_ware,  on='node_id', how='left')
    .merge(nodes_store, on='node_id', how='left', suffixes=('','_st')))
nodes['lat']  = nodes['lat'].fillna(nodes['lat_st'])
nodes['lng']  = nodes['lng'].fillna(nodes['lng_st'])
nodes['type'] = nodes['type'].fillna(nodes['type_st'])
nodes = nodes[['node_id','lat','lng','type']]
nodes.to_csv('nodes.csv', index=False)
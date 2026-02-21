#!/usr/bin/env python3
import os
import pandas as pd

SRC_DIRS = [
    '/root/data/合成/源数据',
    '/root/data/合成/源数据 copy'
]
OUT_DIR = '/root/data/合成'
OUT_FILE = os.path.join(OUT_DIR, 'combined_final.csv')

CITY_COLS = ['city','城市','name','city_name','cityName']
TREATED_COLS = ['treated','is_treated','处理','treated_flag']
POP_COLS = ['registered_population_10k','pop','population','人口']
CONC_COLS = ['c_pm25','pm25','concentration','浓度','C_PM25','PM2.5','C_PM2.5']
TIME_COLS = ['week_end','week','year_week','date','time']

os.makedirs(OUT_DIR, exist_ok=True)

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def read_table(path):
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, low_memory=False)
    else:
        return pd.read_excel(path)

df_list = []
for d in SRC_DIRS:
    if not os.path.isdir(d):
        print(f'skipping missing dir: {d}')
        continue
    for root, _, files in os.walk(d):
        for fn in files:
            if not (fn.lower().endswith('.csv') or fn.lower().endswith(('.xls', '.xlsx'))):
                continue
            path = os.path.join(root, fn)
            try:
                df = read_table(path)
            except Exception as e:
                print(f'跳过无法读取文件: {path} -> {e}')
                continue
            
            df.columns = [str(c).strip() for c in df.columns]
            city_col = find_col(df.columns, CITY_COLS)
            treated_col = find_col(df.columns, TREATED_COLS)
            pop_col = find_col(df.columns, POP_COLS)
            conc_col = find_col(df.columns, CONC_COLS)
            time_col_file = find_col(df.columns, TIME_COLS)
            
            if not all([city_col, treated_col, pop_col, conc_col, time_col_file]):
                continue
            
            # 提取需要的列并统一命名
            df = df[[city_col, treated_col, pop_col, conc_col, time_col_file]].copy()
            df.columns = ['city', 'treated', 'pop', 'conc', 'time']
            df_list.append(df)

if not df_list:
    print('未找到可处理的数据文件，退出。')
    raise SystemExit(1)

all_df = pd.concat(df_list, ignore_index=True)
all_df['city'] = all_df['city'].astype(str).str.strip()
all_df['time'] = pd.to_datetime(all_df['time'], errors='coerce')
all_df['treated'] = pd.to_numeric(all_df['treated'], errors='coerce').fillna(0).astype(int)
all_df['pop'] = pd.to_numeric(all_df['pop'], errors='coerce').fillna(0)
all_df['conc'] = pd.to_numeric(all_df['conc'], errors='coerce')

# 去除完全重复的行（同一城市同一时间的重复记录）
all_df = all_df.drop_duplicates(subset=['city', 'time'])

print(f'总记录数（去重后）：{len(all_df)}')

# 识别“处理组城市”：在任意时间点出现 treated==1 的城市（ever-treated）
ever_treated = all_df.groupby('city', dropna=False)['treated'].max()
treated_cities = ever_treated[ever_treated == 1].index.tolist()

treated_city_rows = all_df[all_df['city'].isin(treated_cities)].copy()
control_rows = all_df[~all_df['city'].isin(treated_cities)].copy()

treated_rows_count = int((treated_city_rows['treated'] == 1).sum())
print(f'处理组城市数(ever-treated): {len(treated_cities)}')
print(f'处理组 treated==1 记录数: {treated_rows_count}')
print(f'控制组城市数(never-treated): {control_rows["city"].nunique()}')
print(f'控制组记录数: {len(control_rows)}')

if treated_city_rows.empty:
    raise SystemExit('没有识别到任何 ever-treated 城市，无法生成“合成”处理组')

t0_time = treated_city_rows.loc[treated_city_rows['treated'] == 1, 'time'].min()
if pd.isna(t0_time):
    raise SystemExit('ever-treated 城市中没有任何 treated==1 的时间点，无法确定政策时点 t0')

print(f'合成序列政策时点 t0 = {pd.Timestamp(t0_time).date()}')

# 对处理组城市按时间进行人口加权平均，生成“合成”城市（覆盖政策前后所有周）
def weighted_mean(group):
    total_pop = group['pop'].sum()
    if total_pop > 0:
        w_conc = (group['pop'] * group['conc']).sum() / total_pop
    else:
        w_conc = group['conc'].mean()
    return pd.Series({
        'city': '合成',
        'pop': total_pop,
        'conc': w_conc
    })

syn_df = treated_city_rows.groupby('time').apply(weighted_mean).reset_index()
syn_df['treated'] = (syn_df['time'] >= pd.Timestamp(t0_time)).astype(int)

# 合并：控制组不变(never-treated 城市) + 合成的处理组(ever-treated 城市聚合)
final_df = pd.concat([control_rows, syn_df], ignore_index=True)

# 重命名 conc -> pm25，并拆分 year/week
final_df = final_df.rename(columns={'conc': 'pm25'})
final_df['time'] = pd.to_datetime(final_df['time'], errors='coerce')
iso = final_df['time'].dt.isocalendar()
final_df['year'] = iso.year.astype(int)
final_df['week'] = iso.week.astype(int)

# 排序：按城市和时间排序
final_df = final_df.sort_values(by=['city', 'time']).reset_index(drop=True)

# 输出列顺序
cols = ['city', 'treated', 'pop', 'pm25', 'time', 'year', 'week']
final_df = final_df[cols]

final_df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print(f'已生成最终文件: {OUT_FILE}')
print(f'最终输出行数: {len(final_df)}')

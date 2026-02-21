import os
import pandas as pd
from collections import defaultdict

SRC_DIRS = [
    '/root/data/合成/源数据',
    '/root/data/合成/源数据 copy'
]
CITY_CANDS = ['city', 'City', 'CITY', '城市', 'city_name']
TREATED_CANDS = ['treated', 'is_treated', 'treatment', 'treated_flag']

found = defaultdict(int)
file_details = []
files_scanned = 0
files_with_treated = 0

for src in SRC_DIRS:
    for root, dirs, files in os.walk(src):
        for fn in files:
            if not fn.lower().endswith(('.csv', '.xls', '.xlsx')):
                continue
            path = os.path.join(root, fn)
            files_scanned += 1
            try:
                if fn.lower().endswith('.csv'):
                    # stream in chunks to avoid missing treated later in file
                    chunk_iter = pd.read_csv(path, chunksize=100000, low_memory=False)
                else:
                    # smallish excel: read fully
                    df_full = pd.read_excel(path)
                    chunk_iter = [df_full]
            except Exception:
                continue
            treated_col = None
            city_col = None
            treated_rows_total = 0
            cities_set = set()
            for chunk in chunk_iter:
                cols = list(chunk.columns)
                if treated_col is None:
                    for c in TREATED_CANDS:
                        if c in cols:
                            treated_col = c
                            break
                if city_col is None:
                    for c in CITY_CANDS:
                        if c in cols:
                            city_col = c
                            break
                if treated_col is None:
                    # cannot detect treated column in this chunk/file
                    continue
                try:
                    mask = chunk[treated_col].isin([1, '1', True, 'True'])
                except Exception:
                    continue
                if mask.any():
                    treated_rows_total += int(mask.sum())
                    if city_col and city_col in chunk.columns:
                        cities_set.update(chunk.loc[mask, city_col].dropna().astype(str).str.strip().unique())
            if treated_rows_total>0:
                files_with_treated += 1
                file_details.append((path, treated_rows_total, sorted(cities_set)))
                for c in cities_set:
                    found[c] += treated_rows_total

# write outputs
sum_out = '/root/data/合成/treated_cities_summary_recheck.csv'
with open(sum_out, 'w', encoding='utf-8-sig') as f:
    f.write('files_scanned,{}\n'.format(files_scanned))
    f.write('files_with_treated,{}\n'.format(files_with_treated))
    f.write('unique_treated_cities,{}\n'.format(len(found)))
    for city,cnt in sorted(found.items(), key=lambda x:-x[1]):
        f.write('"{}",{}\n'.format(city,cnt))

list_out = '/root/data/合成/treated_files_full_list.csv'
with open(list_out, 'w', encoding='utf-8-sig') as f:
    f.write('file,treated_rows,cities\n')
    for path,cnt,cities in file_details:
        f.write('"{}",{},"{}"\n'.format(path,cnt,' | '.join(cities)))

print('扫描完成。')
print('源文件数:', files_scanned)
print('含 treated==1 的文件数:', files_with_treated)
print('不同 treated 城市数:', len(found))
print('汇总已写入:', sum_out)
print('文件清单已写入:', list_out)

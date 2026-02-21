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
                    df = pd.read_csv(path, nrows=2000, low_memory=False)
                else:
                    df = pd.read_excel(path, nrows=2000)
            except Exception:
                continue
            cols = [c for c in df.columns]
            treated_col = None
            city_col = None
            for c in TREATED_CANDS:
                if c in cols:
                    treated_col = c
                    break
            for c in CITY_CANDS:
                if c in cols:
                    city_col = c
                    break
            if treated_col is None or city_col is None:
                continue
            # check any treated==1
            try:
                treated_vals = df[treated_col]
            except Exception:
                continue
            if treated_vals.isin([1, '1', True, 'True']).any():
                files_with_treated += 1
                # read full file to get all city names where treated==1
                try:
                    if fn.lower().endswith('.csv'):
                        df_full = pd.read_csv(path, low_memory=False)
                    else:
                        df_full = pd.read_excel(path)
                except Exception:
                    continue
                if treated_col not in df_full.columns or city_col not in df_full.columns:
                    continue
                mask = df_full[treated_col].isin([1, '1', True, 'True'])
                cities = df_full.loc[mask, city_col].dropna().unique()
                for c in cities:
                    found[str(c).strip()] += 1

# write summary
out_lines = []
out_lines.append(f"files_scanned,{files_scanned}")
out_lines.append(f"files_with_treated,{files_with_treated}")
out_lines.append(f"unique_treated_cities,{len(found)}")
for city, cnt in sorted(found.items(), key=lambda x: -x[1]):
    out_lines.append(f'"{city}",{cnt}')

out_path = '/root/data/合成/treated_cities_summary.csv'
with open(out_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(out_lines))

print('扫描完成。')
print(f'源文件数: {files_scanned}, 含 treated=1 的文件: {files_with_treated}')
print(f'找到 treated=1 的不同城市数量: {len(found)}')
print('摘要已写入:', out_path)

import pandas as pd

file_path = '/root/data/合成/combined_final.csv'
df = pd.read_csv(file_path)

# 重命名 conc 为 pm25
df = df.rename(columns={'conc': 'pm25'})

# 确保 time 是 datetime 类型
df['time'] = pd.to_datetime(df['time'])

# 提取 year 和 week
df['year'] = df['time'].dt.isocalendar().year
df['week'] = df['time'].dt.isocalendar().week

# 重新排列列顺序，把 year 和 week 放在 time 旁边
cols = ['city', 'treated', 'pop', 'pm25', 'time', 'year', 'week']
df = df[cols]

# 保存回文件
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print(f"已更新文件: {file_path}")
print("前 5 行数据预览:")
print(df.head())

""" Classes
0	Nederzettingen en kolonistengeweld
1	Israëlische veiligheidstroepen & bezettingsautoriteiten
2	Internationaal Strafhof         -- nothing for wafa...
3	COVID-19                        -- one hit for wafa (123912) , but not in csv file because under CABINET, not OCCUPATION
4	Rapporten                       -- one hit (124108), but also classified as 1 already
5	Palestijnse verkiezingen
6	Israëlische verkiezingen en formatie    -- nothing for wafa...
7	Overig                          -- one hit (123912) but not in csv file because under POLITICS, not OCCUPATION
"""
# Classified so far: 1-3-2021 t/m 20-4/2021

import pandas as pd
import math


classes_using = [0, 1]
# 0	Nederzettingen en kolonistengeweld
# 1	Israëlische veiligheidstroepen & bezettingsautoriteiten

df = pd.read_csv('Manual Classification/Wafa_classified.csv')
class_col = df.Class
num_classified = class_col.count()
print(f'Number of articles classified: {num_classified} \n'
      f'Out of: {len(class_col)} \n'
      f'Percentage classified: {num_classified / len(class_col)}')

# select subset of data that was classified (including articles that were not given a class - apparently not relevant)
nans = class_col.apply(lambda x: math.isnan(x))
not_nans = nans[~nans]
first_classified, last_classified = not_nans.index[0], not_nans.index[-1]
print(f'Percentage considered when classifying: {(last_classified-first_classified) / len(class_col)}')   # not everything necessarily classified because not relevant

df_classified = df.iloc[first_classified:last_classified+1]
df_not_classified = df.iloc[:first_classified].append(df.iloc[last_classified+1:])

# df2 with only classes classified as 0 or 1, -1 for not classified (irrelevant articles?)
print(f'Percentage irrelevant of those considered when classifying: {sum(df_classified.Class.isna()) / len(df_classified.Class)}')
nans_to_minus_1 = df_classified.Class.apply(lambda x: -1 if math.isnan(x) else x)
df_classified2 = df_classified.copy(deep=True)
df_classified2.Class = nans_to_minus_1

# df3 with only classes classified as 0 or 1, remove irrelevant articles
df_classified3 = df_classified.copy(deep=True)
keep = df_classified3.Class.apply(lambda x: False if (math.isnan(x) or x not in classes_using) else True)
df_classified3 = df_classified3[keep]

# Choosing df_classified3
print('-----------------------------------------')
print(f'Length classified set: {len(df_classified3)} \n'
      f'Length unclassified set: {len(df_not_classified)} \n'
      f'Percentage classified: {len(df_classified3) / len(df_not_classified)}')

# Save for further processing
df_not_classified.to_pickle('Manual Classification/unlabeled.pkl', index=True)
df_classified3.to_pickle('Manual Classification/labeled.pkl')

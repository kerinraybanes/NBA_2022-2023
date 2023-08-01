import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')

#IMPORTING DATA

df_reg = pd.read_csv(r'C:\Users\Kerin B\Desktop\Datasets\nba_regular_2223.csv', delimiter=',')
df_playoffs = pd.read_csv(r'C:\Users\Kerin B\Desktop\Datasets\nba_playoffs_2023.csv', delimiter=',')

#DATA UNDERSTANDING

print(df_reg.shape)
print(df_playoffs.shape)

print(df_reg.head(5))
print(df_playoffs.head(5))

print(df_reg.tail(5))
print(df_playoffs.tail(5))

print(df_reg.columns)
print(df_playoffs.columns)

print(df_reg.dtypes)
print(df_playoffs.dtypes)

print(df_reg.describe)
print(df_playoffs.describe)

#DATA PREPARATION AND CLEANING
#Duplicate Player names due to team changes; remove duplicate name rows and keep rows that have "total" stats
counts = df_reg['Player'].value_counts()
players_to_drop = counts[counts > 1].index
total = 'TOT'
print(df_reg)

filtered_df = df_reg[~(df_reg['Player'].isin(players_to_drop) & (df_reg['Tm'] != total))]
print(filtered_df)

#Verify if Duplicate names have been successfully removed.
number_of_players = filtered_df['Player'].value_counts().sum()
unique_players = df_reg['Player'].nunique()
print('Number of Entries:', number_of_players, '-- Number of Unique Players:', unique_players)
#can also do
duplicates_mask = filtered_df['Player'].duplicated()
duplicates_df = filtered_df[duplicates_mask]
print("Duplicate Rows:")
print(duplicates_df)

filtered_df_new = filtered_df.drop(['Rk', 'Age', 'Tm', 'GS'], axis=1)
filtered_df_new.set_index('Player', inplace = True)
print(filtered_df_new.head(5))

#Creating New Columns
criteria1 = 65
criteria2 = 20
filtered_df_mvp = filtered_df_new['MVP Eligibility'] = np.where((filtered_df_new['G'] >= criteria1) & (filtered_df_new['MP'] >= criteria2), 'Yes', 'No')

#Drop Rows of Players who are not MVP Eligible
df_reg_new = filtered_df_new[filtered_df_new['MVP Eligibility'] != 'No']
print(df_reg_new.head(5))

#Top Scorers
df_reg_pts = df_reg_new['PTS'].sort_values(ascending=False)
df_reg_pts.head(10).plot(kind='bar', title='Top 10 Scorers', xlabel = 'Player', ylabel = 'Points Per Game')
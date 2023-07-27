import pandas as pd
import numpy as np

#Import the two datasets for regular season and playoffs

df_reg = pd.read_csv(r'C:\Users\Kerin B\Desktop\Datasets\nba_regular_2223.csv', delimiter=',')
df_playoffs = pd.read_csv(r'C:\Users\Kerin B\Desktop\Datasets\nba_playoffs_2023.csv', delimiter=',')

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

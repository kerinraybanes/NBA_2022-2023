import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')

#STEP 0: IMPORTING AND READING DATA

df_reg = pd.read_csv(r'C:\Users\Kerin B\Desktop\Datasets\nba_regular_2223.csv', delimiter=',')

#---------------------------------------------------------------------------

#STEP 1: DATA UNDERSTANDING
#Size of Dataframe
print(df_reg.shape)

#First 5 Rows of Dataframe
print(df_reg.head(5))

#Last 5 Rows of Dataframe
print(df_reg.tail(5))

#List of Columns
print(df_reg.columns)

#Value Types for each column
print(df_reg.dtypes)

#---------------------------------------------------------------------------

#STEP 2: DATA PREPARATION AND CLEANING
#Duplicate Player names due to team changes; remove duplicate name rows and keep rows that have "total" stats
counts = df_reg['Player'].value_counts()
players_to_drop = counts[counts > 1].index
total = 'TOT'
print(players_to_drop)

#Filter out Duplicate rows, keeping total stats for that player
filtered_df = df_reg[~(df_reg['Player'].isin(players_to_drop) & (df_reg['Tm'] != total))]
print(filtered_df.shape)

#Verify if Duplicate names have been successfully removed.
number_of_players = filtered_df['Player'].value_counts().sum()
unique_players = df_reg['Player'].nunique()
print('Number of Entries:', number_of_players, '-- Number of Unique Players:', unique_players)

#can also do
duplicates_mask = filtered_df['Player'].duplicated()
duplicates_df = filtered_df[duplicates_mask]
print("Duplicate Rows:")
print(duplicates_df)

#Drop Columns that will not be used for statistical analysis
filtered_df_new = filtered_df.drop(['Rk', 'Age', 'Tm', 'GS'], axis=1)
filtered_df_new.set_index('Player', inplace = True)
print(filtered_df_new.head(5))

#---------------------------------------------------------------------------

#STEP 3: Creating New Columns
#Criteria Required to be met in order to be eligible for MVP Award, create new Yes/No Column for MVP Eligibility
criteria_games = 65
criteria_minutes = 20
filtered_df_mvp = filtered_df_new['MVP Eligibility'] = np.where((filtered_df_new['G'] >= criteria_games) & (filtered_df_new['MP'] >= criteria_minutes), 'Yes', 'No')

#Drop Rows of Players who are not MVP Eligible
df_reg_new = filtered_df_new[filtered_df_new['MVP Eligibility'] != 'No']

#Create New COlumn for Total Offensive and Defensive Stats
df_reg_new['Tot_Off'] = df_reg_new['PTS'] + df_reg_new['AST'] + df_reg_new['ORB']
df_reg_new['Tot_DEF'] = df_reg_new['STL'] + df_reg_new['BLK'] + df_reg_new['DRB']
print(df_reg_new.head(5))

#---------------------------------------------------------------------------

#STEP 4: FEATURE RELATIONSHIPS - DATA EXPLORATION
#Count number of players in each position, display in bar graph.
position_counts = df_reg_new['Pos'].value_counts()

plt.figure(figsize = (8,6))
plt.bar(position_counts.index, position_counts.values, color = 'blue')

plt.xlabel('Position')
plt.ylabel('Number of MVP Eligible Players')
plt.title('Distribution of MVP Eligible Player Positions')
plt.show()

#Top Scorers ==> Top 10 Players of the Regular Season
df_reg_top = df_reg_new.sort_values(by = 'PTS', ascending=False).head(10)
df_reg_top['PTS'].plot(kind='bar', title='Top 10 Scorers', xlabel = 'Player', ylabel = 'Points Per Game')

#Field Goal Efficiency of Top 10 Players
pointsfg = ['PTS','FG', 'FGA', 'FG%', 'eFG%']

ax = df_reg_top.plot(kind = 'scatter', title = 'Field Goal Efficiency of Top Scorers', x = 'FGA', y = 'FG')
for i, row in df_reg_top.iterrows():
    ax.annotate(row.name, (row['FGA'], row['FG']), textcoords="offset points", xytext=(5, -10), ha='center')
    ax.annotate(row['FG%'], (row['FGA'], row['FG']), textcoords="offset points", xytext=(5, -20), ha='center', color = 'red')
    ax.annotate(row['eFG%'], (row['FGA'], row['FG']), textcoords="offset points", xytext=(5, -30), ha='center', color = 'blue')

ax.scatter([], [], marker='o', label='eFG%', color='blue', alpha=1)
ax.scatter([], [], marker='o', label='FG%', color='red', alpha=1)

ax.legend()

plt.show()
print(df_reg_top[pointsfg])

#2 Pointer Efficiency of Top 10 Players
points2 = ['PTS','2P', '2PA', '2P%']

ax = df_reg_top.plot(kind = 'scatter', title = '2 Pointers of Top Scorers', x = '2PA', y = '2P')
for i, row in df_reg_top.iterrows():
    ax.annotate(row.name, (row['2PA'], row['2P']), textcoords="offset points", xytext=(5, -10), ha='center')
    ax.annotate(row['2P%'], (row['2PA'], row['2P']), textcoords="offset points", xytext=(5, -20), ha='center', color = 'red')

ax.scatter([], [], marker='o', label='2P%', color='red', alpha=1)

ax.legend()

plt.show()
print(df_reg_top[points2])

#3 Point Efficiency of Top 10 Players
points3 = ['PTS','3P', '3PA', '3P%']

ax = df_reg_top.plot(kind = 'scatter', title = '3 Pointers of Top Scorers', x = '3PA', y = '3P')
for i, row in df_reg_top.iterrows():
    ax.annotate(row.name, (row['3PA'], row['3P']), textcoords="offset points", xytext=(5, -10), ha='center')
    ax.annotate(row['3P%'], (row['3PA'], row['3P']), textcoords="offset points", xytext=(5, -20), ha='center', color = 'red')

ax.scatter([], [], marker='o', label='3P%', color='red', alpha=1)

ax.legend()

plt.show()
print(df_reg_top[points3])

#Steal and Block vs Fouls of Top 10 Players (Defensive Conversion)
points_STL_BLK_PF = ['PTS', 'STL', 'BLK', 'PF']

ind = np.arange(len(df_reg_top))
width = 0.3
plt.figure(figsize=(20, 6))
plt.bar(ind - width, df_reg_top['STL'], width, label='STL', color='blue')
plt.bar(ind - width, df_reg_top['BLK'], width, label='BLK', color='red', bottom=df_reg_top['STL'])
plt.bar(ind, df_reg_top['PF'], width, label='PF', color='green')

plt.xlabel('Player')
plt.ylabel('Values (Steals, Blocks, Fouls)')
plt.title('Steals and Blocks vs Personal Fouls of Top 10 Players')
plt.xticks(ind, df_reg_top.index)  # Set x-axis labels
plt.legend()
plt.show()

print(df_reg_top[points_STL_BLK_PF])

#Assists vs Turnovers (Non-scoring offensive conversion)
points_tov_ast = ['PTS', 'AST', 'TOV']

ax = df_reg_top.plot(kind = 'scatter', title = 'Assists vs Turnovers of Top Scorers', x = 'TOV', y = 'AST')
for i, row in df_reg_top.iterrows():
    ax.annotate(row.name, (row['TOV'], row['AST']), textcoords="offset points", xytext=(5, -10), ha='center')

plt.show()
print(df_reg_top[points_tov_ast])

#Offensive and Defensive Rebound Contribution
rebounds = ['ORB', 'DRB', 'TRB']

ind = np.arange(len(df_reg_top))
width = 0.3
plt.figure(figsize=(20, 6))
plt.bar(ind - width, df_reg_top['ORB'], width, label='ORB', color='blue')
plt.bar(ind - width, df_reg_top['DRB'], width, label='DRB', color='red', bottom=df_reg_top['ORB'])

plt.xlabel('Player')
plt.ylabel('Number of Rebounds')
plt.title('Offensive and Defensive Rebounds of Top 10 Players')
plt.xticks(ind, df_reg_top.index)  # Set x-axis labels
plt.legend()
plt.show()

print(df_reg_top[rebounds])

#Total Offensive and Defensive Stats of Top Players
off_def_stats = ['Tot_OFF', 'Tot_DEF']

ax = df_reg_top.plot(kind = 'scatter', title = 'Total Offensive and Defensive Stats of Top Players', x = 'Tot_DEF', y = 'Tot_OFF')
for i, row in df_reg_top.iterrows():
    ax.annotate(row.name, (row['Tot_DEF'], row['Tot_OFF']), textcoords="offset points", xytext=(5, -10), ha='center')

plt.show()

print(df_reg_top[off_def_stats])

#Step 5: Saving the Data to csv Files
df_reg_top = df_reg_top.reset_index()
df_reg_top.to_csv('nba_regular_stats_top.csv', index = False)

df_reg_new = df_reg_new.reset_index()
df_reg_new.to_csv('nba_regular_stats_mvp_eligible.csv', index = False)
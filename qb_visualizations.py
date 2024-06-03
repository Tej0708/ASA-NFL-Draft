

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('passer.csv')

df = df.drop(['passDirection', 'passDepth'], axis=1) # drop the columns passDirection and passDepth



df = df[df['passNull'] != 1] # remove all the rows where passNull == 1

df = df[(df['passSack'] != 1) & (df['passAtt'] != 0)] # remove all the rows where passSack == 1 and passAtt = 0

df = df[df['passPosition'] == 'QB'] # only include rows where passPosition is QB

df['Completions'] = np.where(df['passOutcomes'] == 'complete', 1, 0) # create a new column called Completions where the value is 1 if passOutcomes is complete and 0 otherwise
df['Incompletions'] = np.where(df['passOutcomes'] == 'incomplete', 1, 0) # create a new column called Incompletions where the value is 1 if passOutcomes is incomplete and 0 otherwise

df = df.groupby('playerId').agg({'Completions': 'sum', 'Incompletions': 'sum', 'passLength': 'sum', 'passTd': 'sum', 'passInt': 'sum'}).reset_index() # group by playerId and sum the values of Completions, Incompletions, passLength, passTd and passInt

print(df.shape)

df = df[df['passLength'] > 0] # remove all the rows where passLength > 0
df['passAtt'] = df['Completions'] + df['Incompletions']
df['CompletionPercentage'] = df['Completions'] / (df['Completions'] + df['Incompletions']) * 100

df = df[df['passAtt'] > 30]

df['passLengthPerAttempt'] = df['passLength'] / df['passAtt']

df['passLengthPerCompletion'] = df['passLength' ] / df['Completions']


df['InterceptionPercentage'] = df['passInt'] / df['passAtt'] * 100



df['passTdPercentage'] = df['passTd'] / df['passAtt'] * 100

plt.scatter(df['Completions'], df['passLength']) # scatter plot of Completions vs passLength
plt.xlabel('Completions')
plt.ylabel('passLength')
plt.tight_layout()
plt.show()


correlation = df[['passLengthPerCompletion', 'Incompletions', 'passTd', 'passInt']].corr() # correlation matrix
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', )
plt.show()


df_sorted = df.sort_values(by=['passTd'], ascending=[False])
top_50 = df_sorted.head(50) # get the top 50 rows
bottom_50 = df_sorted.tail(50) # get the bottom 50 rows

top_50['Group'] = 'Top 50' # create a new column called Group and set the value to 'Top 50'
bottom_50['Group'] = 'Bottom 50' # create a new column called Group and set the value to 'Bottom 50'
combined_df = pd.concat([top_50, bottom_50]) # concatenate the top 50 and bottom 50 dataframes

metrics = ['CompletionPercentage', 'passLengthPerCompletion' ] # list of metrics to compare

sns.set_style('whitegrid')
pallete = {'Top 50': 'royalblue', 'Bottom 50': 'salmon'}
plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, 2, i+1)
    sns.boxplot(x='Group', y=metric, data=combined_df, palette=pallete, showfliers=False)
    plt.title(metric, fontsize=14)
    plt.grid(True, which='both', linestyle='--', lw=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    
    
plt.gcf().set_size_inches(20, 10)
   
plt.suptitle('Comparison of Top 50 and Bottom 50 Quarterbacks', fontsize=20, weight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3 )
plt.show()

sns.kdeplot(top_50['passLengthPerCompletion'], color='royalblue', label='Top 50', fill = True , alpha=0.5)
sns.kdeplot(bottom_50['passLengthPerCompletion'], color='salmon', label='Bottom 50', alpha=0.5, fill = True)
plt.xlabel('Passing Yards Per Completion')
plt.ylabel('Percentage of Quarterbacks')
plt.title('Density of Yards Per Completion for Top 50 and Bottom 50 Quarterbacks', fontsize=14, weight='bold')

plt.show()


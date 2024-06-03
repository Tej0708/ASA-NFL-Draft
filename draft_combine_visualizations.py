import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the Downloads folder
downloads_folder = os.path.expanduser('~/Downloads')

# Load the CSV files
draft_df = pd.read_csv(os.path.join(downloads_folder, 'draft.csv'))
combine_df = pd.read_csv(os.path.join(downloads_folder, 'combine.csv'))

# Identifying common years
common_years = set(draft_df['draft']).intersection(set(combine_df['combineYear']))

# Filtering datasets by common years
draft_df_filtered = draft_df[draft_df['draft'].isin(common_years)]
combine_df_filtered = combine_df[combine_df['combineYear'].isin(common_years)]

# Merging the datasets on `playerId`
merged_df = pd.merge(draft_df_filtered, combine_df_filtered, on='playerId')

# Selecting and formatting relevant columns with corrected names
relevant_columns = [
    'playerId', 'nameFirst_x', 'nameLast_x', 'draft', 'combineYear', 'position_x', 'combinePosition',
    'combineHeight', 'combineWeight', 'combineHand', 'pick'
]
cleaned_df = merged_df[relevant_columns]

# Rename columns for clarity
cleaned_df.columns = [
    'playerId', 'firstName', 'lastName', 'draftYear', 'combineYear', 'draftPosition', 'combinePosition',
    'height', 'weight', 'handSize', 'draftPick'
]

# Sort the resulting dataframe by `playerId`
cleaned_df = cleaned_df.sort_values(by='playerId').reset_index(drop=True)

# Save the cleaned dataframe to a new CSV file in the Downloads folder
output_path = os.path.join(downloads_folder, 'cleaned_draft_combine_data.csv')
cleaned_df.to_csv(output_path, index=False)

print(f"Cleaned data has been saved to {output_path}")

# Remove general positions if specific ones are present
exclude_positions = ['LB', 'OL', 'DL', 'DB', 'PJ', 'LS', 'EDG', 'S']
specific_positions = ['ILB', 'OLB', 'OT', 'OG']
filtered_df = cleaned_df[~cleaned_df['combinePosition'].isin(exclude_positions) | cleaned_df['combinePosition'].isin(specific_positions)]

# Further filter to exclude specific positions DL, DB, PJ, LS, EDG, and S
filtered_df = filtered_df[~filtered_df['combinePosition'].isin(['DL', 'DB', 'PK', 'LS', 'EDG', 'S'])]

# Set up the plotting environment
sns.set(style="whitegrid")

# Combine height per position
plt.figure(figsize=(14, 7))
sns.boxplot(x='combinePosition', y='height', data=filtered_df, palette="viridis")
plt.title('Combine Height per Position', fontsize=16, weight='bold')
plt.xlabel('Position', fontsize=14)
plt.ylabel('Height (inches)', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Hand size per position
plt.figure(figsize=(14, 7))
sns.boxplot(x='combinePosition', y='handSize', data=filtered_df, palette="viridis")
plt.title('Hand Size per Position', fontsize=16, weight='bold')
plt.xlabel('Position', fontsize=14)
plt.ylabel('Hand Size (inches)', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Height by year
plt.figure(figsize=(14, 7))
sns.lineplot(x='draftYear', y='height', data=filtered_df, marker='o', color='b')
plt.title('Height by Draft Year', fontsize=16, weight='bold')
plt.xlabel('Draft Year', fontsize=14)
plt.ylabel('Height (inches)', fontsize=14)
plt.show()

# Weight by year
plt.figure(figsize=(14, 7))
sns.lineplot(x='draftYear', y='weight', data=filtered_df, marker='o', color='r')
plt.title('Weight by Draft Year', fontsize=16, weight='bold')
plt.xlabel('Draft Year', fontsize=14)
plt.ylabel('Weight (lbs)', fontsize=14)
plt.show()

# Weight per position
plt.figure(figsize=(14, 7))
sns.boxplot(x='combinePosition', y='weight', data=filtered_df, palette="viridis")
plt.title('Weight per Position', fontsize=16, weight='bold')
plt.xlabel('Position', fontsize=14)
plt.ylabel('Weight (lbs)', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Draft pick distribution by position
plt.figure(figsize=(14, 7))
sns.histplot(data=filtered_df, x='draftPick', hue='combinePosition', multiple='stack', bins=30, palette="viridis", element="step", stat="count")
plt.title('Draft Pick Distribution by Position', fontsize=16, weight='bold')
plt.xlabel('Draft Pick', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

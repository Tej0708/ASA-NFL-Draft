import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from PIL import Image


# Set page configuration
st.set_page_config(
    page_title="Custom Color Palette",
    page_icon=":paintbrush:",
    layout="wide",  # 'wide' layout
    initial_sidebar_state="expanded",  # Expanded sidebar
)
# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('/Users/tejgaonkar/Downloads/cpd_cleaned.csv')
def load_passer():
    # Function to load and preprocess data
    df = pd.read_csv('/Users/tejgaonkar/Downloads/passer.csv')
    df = df.drop(['passDirection', 'passDepth'], axis=1)
    df = df[df['passNull'] != 1]
    df = df[(df['passSack'] != 1) & (df['passAtt'] != 0)]
    df = df[df['passPosition'] == 'QB']
    df['Completions'] = np.where(df['passOutcomes'] == 'complete', 1, 0)
    df['Incompletions'] = np.where(df['passOutcomes'] == 'incomplete', 1, 0)
    df = df.groupby('playerId').agg({'Completions': 'sum', 'Incompletions': 'sum', 'passLength': 'sum', 'passTd': 'sum', 'passInt': 'sum'}).reset_index()
    df = df[df['passLength'] > 0]
    df['passAtt'] = df['Completions'] + df['Incompletions']
    df['CompletionPercentage'] = df['Completions'] / (df['Completions'] + df['Incompletions']) * 100
    df = df[df['passAtt'] > 30]
    df['passLengthPerAttempt'] = df['passLength'] / df['passAtt']
    df['passLengthPerCompletion'] = df['passLength'] / df['Completions']
    df['InterceptionPercentage'] = df['passInt'] / df['passAtt'] * 100
    df['passTdPercentage'] = df['passTd'] / df['passAtt'] * 100
    return df
def load_passer_graphs(df):
    # Function to create and display plots
    st.subheader('Scatter Plot: Completions vs Pass Length')
    fig, ax = plt.subplots()
    ax.scatter(df['Completions'], df['passLength'])
    ax.set_xlabel('Completions')
    ax.set_ylabel('Pass Length')
    ax.set_title('Completions vs. Pass Length in NFL QBs')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader('Correlation Heatmap')
    correlation = df[['passLengthPerCompletion', 'Incompletions', 'passTd', 'passInt']].corr()
    fig, ax = plt.subplots()
    ax.set_title('Correlation Heatmap for QB statistics')
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', ax=ax)
    st.pyplot(fig)
    
    df_sorted = df.sort_values(by=['passTd'], ascending=[False])
    top_50 = df_sorted.head(50)
    bottom_50 = df_sorted.tail(50)
    
    top_50['Group'] = 'Top 50'
    bottom_50['Group'] = 'Bottom 50'
    combined_df = pd.concat([top_50, bottom_50])
    
    metrics = ['CompletionPercentage', 'passLengthPerCompletion']
    pallete = {'Top 50': 'royalblue', 'Bottom 50': 'salmon'}
    
    st.subheader('Comparison of Top 50 and Bottom 50 Quarterbacks')
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    pallete = {'Top 50': 'royalblue', 'Bottom 50': 'salmon'}
    for i, metric in enumerate(metrics):
        sns.boxplot(x='Group', y=metric, data=combined_df, palette=pallete, showfliers=False, ax=axes[i])
        axes[i].set_title(metric, fontsize=14)
        axes[i].grid(True, which='both', linestyle='--', lw=0.7)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    fig.suptitle('Comparison of Top 50 and Bottom 50 Quarterbacks', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    st.pyplot(fig)
    
    st.subheader('Density of Yards Per Completion for Top 50 and Bottom 50 Quarterbacks')
    fig, ax = plt.subplots()
    sns.kdeplot(top_50['passLengthPerCompletion'], color='royalblue', label='Top 50', fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(bottom_50['passLengthPerCompletion'], color='salmon', label='Bottom 50', fill=True, alpha=0.5, ax=ax)
    ax.set_xlabel('Passing Yards Per Completion')
    ax.set_ylabel('Percentage of Quarterbacks')
    ax.set_title('Density of Yards Per Completion for Top 50 and Bottom 50 Quarterbacks', fontsize=14, weight='bold')
    ax.legend()
    st.pyplot(fig)

    st.write("""
    Our key insights given by an examination of quarterbacks ranked by total passing yards reveals 
    a potential correlation between career success and a preference for short-to-intermediate 
    throws. This finding suggests that quarterbacks who consistently generate high yardage totals 
    may prioritize lower-risk passing strategies, potentially emphasizing checkdowns and efficient 
    completions.
             
    Building on the initial observation, a deeper analysis could explore the prevalence of low-to-mid-
    risk passing strategies amongst top quarterbacks. This might reveal a significantly higher 
    adoption rate of such strategies compared to lower-ranked players. Conversely, some lower-
    ranked quarterbacks might exhibit an overemphasis on caution, potentially neglecting the 
    effectiveness of the 8-18 yard passing range historically associated with success.
             
    What does this mean for NFL Teams? Contrary to what most people think, the quarterbacks who were more conservative with the ball,
    essentially those who were careful with the football and limited big plays and turnovers, are the most sucessful in the pros. This
    could mean that QB's with college tape that shows them being more careful with the football might have more success and might be a trait
    of a franchise QB. NFL Teams should definitely take this more into account when drafting their next QB!


    """)

def load_draft_combine():
    # Load the data
    draft_df = pd.read_csv('draft.csv')
    combine_df = pd.read_csv('combine.csv')

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

    # Remove general positions if specific ones are present
    exclude_positions = ['LB', 'OL', 'DL', 'DB', 'PJ', 'LS', 'EDG', 'S']
    specific_positions = ['ILB', 'OLB', 'OT', 'OG']
    filtered_df = cleaned_df[~cleaned_df['combinePosition'].isin(exclude_positions) | cleaned_df['combinePosition'].isin(specific_positions)]

    # Further filter to exclude specific positions DL, DB, PJ, LS, EDG, and S
    filtered_df = filtered_df[~filtered_df['combinePosition'].isin(['DL', 'DB', 'PK', 'LS', 'EDG', 'S'])]

    # Set up the plotting environment
    sns.set(style="whitegrid")

    # Combine height per position
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='combinePosition', y='height', data=filtered_df, palette="viridis", ax=ax1)
    ax1.set_title('Combine Height per Position', fontsize=16, weight='bold')
    ax1.set_xlabel('Position', fontsize=14)
    ax1.set_ylabel('Height (inches)', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Hand size per position
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='combinePosition', y='handSize', data=filtered_df, palette="viridis", ax=ax2)
    ax2.set_title('Hand Size per Position', fontsize=16, weight='bold')
    ax2.set_xlabel('Position', fontsize=14)
    ax2.set_ylabel('Hand Size (inches)', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Height by year
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    sns.lineplot(x='draftYear', y='height', data=filtered_df, marker='o', color='b', ax=ax3)
    ax3.set_title('Height by Draft Year', fontsize=16, weight='bold')
    ax3.set_xlabel('Draft Year', fontsize=14)
    ax3.set_ylabel('Height (inches)', fontsize=14)
    st.pyplot(fig3)

    # Weight by year
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    sns.lineplot(x='draftYear', y='weight', data=filtered_df, marker='o', color='r', ax=ax4)
    ax4.set_title('Weight by Draft Year', fontsize=16, weight='bold')
    ax4.set_xlabel('Draft Year', fontsize=14)
    ax4.set_ylabel('Weight (lbs)', fontsize=14)
    st.pyplot(fig4)

    # Weight per position
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='combinePosition', y='weight', data=filtered_df, palette="viridis", ax=ax5)
    ax5.set_title('Weight per Position', fontsize=16, weight='bold')
    ax5.set_xlabel('Position', fontsize=14)
    ax5.set_ylabel('Weight (lbs)', fontsize=14)
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

def load_sacks():

    # Load the CSV files
    sacks_df = pd.read_csv('sacks.csv')
    combine_df = pd.read_csv('combine.csv')

    # Clean the datasets to include only the players in both datasets
    common_players = pd.merge(sacks_df[['playerId']], combine_df[['playerId']], on='playerId')
    filtered_sacks_df = sacks_df[sacks_df['playerId'].isin(common_players['playerId'])]
    filtered_combine_df = combine_df[combine_df['playerId'].isin(common_players['playerId'])]

    # Extract relevant columns for EDA and merge datasets on playerId
    relevant_combine_columns = [
        'playerId', 'combineHeight', 'combineWeight', 'combineBench', 
        'combineShuttle', 'combineVert', 'combineBroad', 'combine40yd', 
        'combineArm', 'combineHand', 'combine3cone', 'combine60ydShuttle'
    ]
    filtered_combine_df = filtered_combine_df[relevant_combine_columns]
    relevant_sack_columns = ['playerId', 'sackType', 'sackYards']
    filtered_sacks_df = filtered_sacks_df[relevant_sack_columns]
    merged_df = pd.merge(filtered_combine_df, filtered_sacks_df, on='playerId')

    # Function to plot histograms for combine stats
    def plot_histograms(df):
        combine_stats = [
            'combineHeight', 'combineWeight', 'combineBench', 'combineShuttle', 
            'combineVert', 'combineBroad', 'combine40yd', 'combineArm', 
            'combineHand', 'combine3cone', 'combine60ydShuttle'
        ]
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
        axes = axes.flatten()

        for i, stat in enumerate(combine_stats):
            axes[i].hist(df[stat].dropna(), bins=20, edgecolor='black')
            axes[i].set_title(f'Distribution of {stat}')
            axes[i].set_xlabel(stat)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

    # Function to plot scatter plots for Height, Weight, Bench Press vs. Sack Yards
    def plot_scatter_plots(df):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        axes[0].scatter(df['combineHeight'], df['sackYards'], alpha=0.6)
        axes[0].set_title('Height vs. Sack Yards')
        axes[0].set_xlabel('Height (inches)')
        axes[0].set_ylabel('Sack Yards')

        axes[1].scatter(df['combineWeight'], df['sackYards'], alpha=0.6)
        axes[1].set_title('Weight vs. Sack Yards')
        axes[1].set_xlabel('Weight (lbs)')
        axes[1].set_ylabel('Sack Yards')

        axes[2].scatter(df['combineBench'], df['sackYards'], alpha=0.6)
        axes[2].set_title('Bench Press vs. Sack Yards')
        axes[2].set_xlabel('Bench Press (reps)')
        axes[2].set_ylabel('Sack Yards')

        plt.tight_layout()
        st.pyplot(fig)

    # Function to plot the correlation matrix
    def plot_correlation_matrix(df):
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    # Streamlit app structure
    st.title("NFL Combine and Sack Analysis")

    # Plot histograms
    st.subheader("Combine Stats Distributions")
    plot_histograms(merged_df)

    # Plot scatter plots
    st.subheader("Height, Weight, Bench Press vs. Sack Yards")
    plot_scatter_plots(merged_df)

    # Plot correlation matrix
    st.subheader("Correlation Matrix")
    plot_correlation_matrix(merged_df)

# Load the ML model
@st.cache_data
def load_model():
    return joblib.load('/Users/tejgaonkar/Downloads/model_f.pkl')

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.write("Select a page to view:")
page = st.sidebar.selectbox("Go to", ["Home",'Draft & Combine Analysis', "Quarterback Passing Analysis", 'Defensive Sacks Analysis', "QB Career Prediction"])

# Your Streamlit app content goes here...

# Home Page
if page == "Home":
    logo_path = '/Users/tejgaonkar/Downloads/ASA_Logo.png'
    image_path = '/Users/tejgaonkar/Downloads/NFL_Draft_logo.png'
    logo = Image.open(logo_path)
    nfl_draft_logo = Image.open(image_path)
    nfl_draft_pic = Image.open('/Users/tejgaonkar/Downloads/NFL_Draft_Picture.png')
    st.image(logo, width = 100)
    
    glow_style = "text-align: center; color: white; text-shadow: 0 0 10px #800080, 0 0 20px #800080, 0 0 30px #800080;"


    st.markdown("<h1 style='{}'>NFL Draft Analysis</h1>".format(glow_style), unsafe_allow_html=True)
    st.markdown("<h2 style='{}'>A Data Driven Approach to Drafting Your Team's Next Star!</h2>".format(glow_style), unsafe_allow_html=True)
    st.image('https://www.statesman.com/gcdn/authoring/authoring-images/2024/04/23/PDTF/73427636007-gl-3-rb-b-0-wyaax-noj.jpeg', width = 1000)
    st.write("""
    
    ## The 2024 Draft is officially open...
    
    The NFL Draft is a defining moment for the young prospects who finally live out their dreams of becoming a pro player. It's just the
    start to a hopefully promising career for them. However, the NFL is a brutal game, not only in the physicality, but in the mental
    strength it takes to perform well on day 1. As much as we would want every prospect to pan out, that is simply always not the case.
    A player might not be drafted to the right situation, making them lose out on a lot of potential money, the front office
    might be fired and fans will be in misery for years to come.
    
    So if you are in the front office or a hardcore fan, you want to draft the right players to put your franchise on 
    the right tracks. Using Aggie Sports Analytics unique data driven approach to the NFL draft, we will give our insights and a solution
    to make sure your team drafts the right player!
    
    ### To explore more:
    - Use the sidebar to navigate between the home page, our data analysis and custom insights for combine & draft, QB's and Defensive Lineman, and our cutting edge prediction algorithm for QB's careers.
             
    
    # Meet the Team:
    """)
    st.image("https://ca.slack-edge.com/T05BWALNT61-U063PFFCBEX-da5f5838a978-72")
    st.write("Tej Gaonkar: Project Manager")
    rahul = Image.open('/Users/tejgaonkar/Downloads/Screenshot 2024-05-29 at 2.37.37 PM.png')
    st.image(rahul, width = 100)
    st.write('Rahul Padhi: Developer')
    st.image("https://ca.slack-edge.com/T05BWALNT61-U06PQG6GX9V-a438c811995e-72")
    st.write("Ahmed Seyam: Developer")
    st.image('https://ca.slack-edge.com/T05BWALNT61-U06PJUHUAPR-dbe3f6f3c1f6-72')
    st.write("Harsh Karuturi: Developer")
    st.image("https://ca.slack-edge.com/T05BWALNT61-U063W1CQY0P-1cecd64e3d8f-72")
    st.write("Luke Harrell: Business")
    st.image('https://ca.slack-edge.com/T05BWALNT61-U064BER9K6X-1df04421c6d9-72')
    st.write('Prabhjot: Media')
    st.image('https://ca.slack-edge.com/T05BWALNT61-U064CMJ7SRG-4ffcbad08e74-72')
    st.write('Devin Seberino: Media')
# Data Analysis Page
elif page == "Quarterback Passing Analysis":
    st.title('Quarterback Passing Analysis')
    st.subheader('Analysis and Graphs done by Ahmed Seyam')
    df = load_passer()
    st.write(f'Data shape: {df.shape}')
    st.dataframe(df.head()) # Display the first few rows of the dataframe

    # Display plots
    load_passer_graphs(df)

    # Add more data analysis and visualizations as needed
elif page == 'Draft & Combine Analysis':
    st.title('Draft & Combine Analysis')
    st.subheader('Analysis and Graphs done by Rahul Padhi')
    load_draft_combine()

elif page == 'Defensive Sacks Analysis':
    st.title('Defensive Sacks Analysis')
    load_sacks()

# ML Model Prediction Page
else:
    st.title("QB Boom or Bust Prediction")
    st.write("## Do you think your next QB will be a star or a bust?")

    st.write("""
    Fill out the QB's information below to get a prediction.
    """)

    # Load the model
    model = load_model()

    team_array = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV',
                  'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
    
    warnings.filterwarnings("ignore", message="NumberInput value below has type int so is displayed as int despite format string %.1f")

    # Input fields for the player data
    ageAtDraft = st.number_input("Age", min_value=18, max_value=40, value=22)
    college = st.selectbox("College", options=['Alabama', 'Clemson', 'Ohio State', 'Michigan',  'LSU', 'Oklahoma', 'Other'])
    round = st.number_input("Draft Round", value = float(0.0), format = '%.1f', step=0.1)
    combineHeight = st.number_input("Height in inches (e.g 74.3)", step = 1, format = '%.1f')
    combineWeight = st.number_input("Weight in lbs (e.g. 213)", min_value = 100, max_value = 400)
    nameFull = st.text_input("First and Last Name of the Prospect")
    combine40yd = st.number_input("40 yd dash time at combine", format = '%.2f')
    combineShuttle = st.number_input('Shuttle drill time at combine', format = '%.2f')
    combineBroad = st.number_input('Broad jump distance at combine', format = '%.2f')
    combine3cone = st.number_input('3 cone drill time at combine', format = '%.2f')
    pick = st.number_input('Draft Pick Number (e.g. 2 or 175)', min_value = 1, value = 10)
    draftTradeValue = st.number_input('Given the chart below, enter the trade value of the pick that your player was picked at:', format = '%.2f')
    draftTeam = st.selectbox("Drafted team", options = team_array)

    # Create a feature vector based on user input
    input_features = np.array([[combineHeight, combineWeight, nameFull, college, ageAtDraft, combine40yd, combineShuttle, combineBroad, combine3cone, round, pick, draftTradeValue, draftTeam]])

    input_features = np.ravel(input_features)

    # Encode categorical features (this example assumes one-hot encoding or label encoding is required)
    # You need to match the encoding with what was used during training
    # For simplicity, this example does not include the actual encoding process
    enc = LabelEncoder()
    input_encoded = enc.fit_transform(input_features)

    input_encoded = input_encoded.reshape(1, -1)
    # Make a prediction
    if st.button("Predict"):
        prediction = model.predict(input_encoded)
        if prediction == 'Star':
            st.write("Your team drafted their next franchise Quarterback!")
        elif prediction == 'Meh':
            st.write("Your team drafted an ok starter.")
        else:
            st.write('Your team drafted a bust, better luck next time :(')
    image = Image.open('/Users/tejgaonkar/Downloads/draft_value_chart.jpg')
    st.image(image, width = 1000)



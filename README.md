# NFL Draft Analysis: A Data-Driven Approach at Drafting your next Superstar!

So many NFL teams keep drafting busts in the draft, setting their franchise back so many years. However, with our data-driven insights, you will never draft one ever again!

## Description

The 2024 Draft is officially open...

The NFL Draft is a defining moment for the young prospects who finally live out their dreams of becoming a pro player. It's just the
start of a hopefully promising career for them. However, the NFL is a brutal game, not only in the physicality but in the mental
strength it takes to perform well on day 1. As much as we would want every prospect to pan out, that is simply always not the case.
A player might not be drafted to the right situation, making them lose out on a lot of potential money, the front office
might be fired and fans will be in misery for years to come.
    
So if you are in the front office or a hardcore fan, you want to draft the right players to put your franchise on 
the right track. Using Aggie Sports Analytics' unique data-driven approach to the NFL draft, we will give our insights and a solution
to make sure your team drafts the right player!

The first 3 sections of our website: Draft and Combine Analysis, QB Passing Analysis, and Defensive Sacks Analysis give teams a comprehensive visual understanding of how to draft players.
Teams can look at past drafted players and some of the key attributes that made them sucessful in the league. This might help them find players with similar measurables and draft those players who will hopefully develop into their next superstar!

Our QB Boom or Bust Prediction uses a Random Forest Algorithm to predict whether your player will be a boom or a bust. Some of the features it takes in are the players Age, College, Combine measurables, etc. We created certain categories to define what a star, meh, or bust quarterback is. Usually those quarterbacks who have a high passing yard career over their career are stars. However this is not 100% accurate but for simplicity we decided to define it like that.

## Getting Started

### Dependencies

#### Python:
* Our application is programmed through streamlit, a python library, so a Python interpreter needs to be installed.
* We would recommend installing Python and Anaconda/VS Code

#### Data:
* For this application, we used data from Kaggle. There are several datasets that will be linked but specifically you will need 'draft.csv', 'combine.csv', 'passer.csv', and 'sacks.csv'
* Here is the link to access the data: [Kaggle Link](https://www.kaggle.com/datasets/toddsteussie/nfl-play-statistics-dataset-2004-to-present)
* Note: you will also need 'passing.csv' if you want to look at the model-building process. That is a file that we created using aggregate commands in Postgres (attached in the .sql file).

#### Model:
* The model is packaged in the 'model_f.pkl' file. You will need to import this and change the file path in your code so the program can find it.

### Installing

* To launch our web application, you will first need to install a bunch of libraries
* To do this you can use the pip install python command. Here is an example for the pandas library
```
pip install pandas
```
* You need to install the streamlit, pandas, numpy, matplotlib, seaborn, sklearn, joblib, warnings, and PIL libraries using the same format as above.

### Executing program

* To run the program you need to enter some commands through the terminal
* To do this you can use the streamlit run command below:
```
streamlit run [pathname]
```
* For example with the pathname of our machine
```
streamlit run /Users/tejgaonkar/Downloads/nfl_draft_frontend.py
```
* Alternatively, you could change your directory to one that contains the python file and run it directly from there
```
cd [directory]
```
```
streamlit run nfl_draft_frontend.py
```
* Here is an example from our machine
```
cd Downloads
```
```
streamlit run nfl_draft_frontend.py
```

## Contributors
Tej Gaonkar (Project Manager): [tagaonkar@ucdavis.edu](mailto:tagaonkar@ucdavis.edu)
Rahul Padhi (Developer):[rpadhi@ucdavis.edu](mailto:rpadhi@ucdavis.edu)
Ahmed Seyam (Developer):[aaseyam@ucdavis.edu](mailto:aaseyam@ucdavis.edu)
Harshith Karuturi (Developer):[hckaruturi@ucdavis.edu](mailto:hckaruturi@ucdavis.edu)

## Acknowledgments
* Project completed under [Aggie Sports Analytics](https://aggiesportsanalytics.com/) at UC Davis.

# 180B

# Applying Post Prediction Inference to the NFL: How We Strive to Reinvent Sports Analysis
 
 > By: Jonathan Langley, Sujeet Yeramareddy, and Yong Liu

### Introduction

For our 180B project, we decided to use our domain methodology of postpi (post prediction inference) and apply it to sports analysis based on NFL games.  We are designing a model that can predict the outcome of a football game, such as which team will win and what the margin of their victory will be, and then correcting the statistical inference for selected key features. The main goals of our investigation is discerning which features most strongly determine the victor of a football game, and subsequently which features provide the most significant means of inferring the margin of that victory. For example, does the home field advantage give a 50% higher chance of winning by 7 points?  Is the comparative offense to defense rating the most critical factor in securing a win?  Does or does not weather play a statistically significant part in influencing margin of victory?  These are just some of the questions we have brought up and seek to answer during the course of our research, and by conducting this project we are attempting to revolutionize the way NFL analytics are conducted via a more accurate statistical method of inference, postpi. 

### Data Collection
Data source: https://stathead.com/football/

For the data collection portion we used this website to manually copy and paste NFL game data from the 2000 season to 2021 season for each individual feature. This process was quite tedious because we were only able to access 100 rows of the data at a time. Additionally, we had to obtain the data for many game features for all NFL games therefore it took longer than expected and required us to split up the simple, yet time-consuming, task. The features that we included in our dataset included, but are not limited to, first downs, passing, rushing, defense, penalties, and temperature. 

### Data Cleaning

We used game characteristics such as teams playing, date of game, week of season, and more to merge the many individual datasets together so that we have one big dataset that included over 60 columns representing 5600+ NFL games in the past 21 years. It is important to note that our dataset actually includes over 10,000 rows because each game must be represented by 2 entries to avoid model bias in Spread prediction. The dataset contains two columns that denotes which NFL team is categorized as "Tm" and "Opp" which represent the order of each teams statistics for the team. By swapping the team in these columns we can represent the same game twice which allows us to include positive and negative spreads in our dataset for our model to learn on.

Once obtaining this large dataset, our next task was to clean our dataset so that our data is easier to explore, analyze, and use to predict. The first step we took was to fix specific columns in our dataset such as "Home" and "Result". The "Home" column originally had an @ symbol when it was a Home game for the team in the "Opp" column, and missing otherwise, therefore we chose to replace the @ values with a 0 to transform this to a binary column that represents when the team in the "Tm" columns was home. Next, the "Result" column was originally in the format of a string like "W 12 - 9" which represents the score of the NFL game in the format "TmOutcome TmScore - OppScore" based on the teams in each of these columns. We took this column and used string manipulation to convert it into a column named Spread which is our response variable that we are trying to predict. Our next step in preprocessing was to impute missing values in columns that contained missing data. We did this by randomly sampling from the column with missing values adn randomly placing them in the column. Most of our columns had a relatively low amount of missing values, therefore by doing this we are not compromising the integrity of our data. Below is a heatmap representing which columns were imputed in our dataset. Our final step in data cleaning was to remove outliers. We did this by using the common rule of thumb which 1.5 times the Interquartile Range (Quartile3 - Quartile1). This is reasonable because it is very uncommon for an NFL game to end with a Spread larger than 35 points, therefore keeping this data is doing nothing more than adding bias to our model. Below we can see that by using this rule, we are removing a minimal amount of data.

### Exploratory Data Analysis

In our exploratoy data analysis, we created histograms of each of our variables to understand any skews in our explanatory variables. Additionally we made scatterplots with each of our variables and our response variable, Spread to see if we can observe any strong relationships prior to building our model. Most of our variables were non-linearly related with Spread and required a model that can find interactions between variables in our data in order to accurately predict Spread. However, the performance of the QB on both teams showed the most correlation with Spread when analyzing the explanatory variables individually with Spread. In the plots below, we can see that the higher the QB rating of the team in the "Tm" column, the Spread tends to favor them, and vice versa. The worse the "Tm" QB does, the spread favors the "Opp". Using this information we are able to understand that this will be an important feature in our model.


### Building Neural Network ML Model
 Through testing ,we finalized our N-N (MLPregressor) model with 4 hidden layer and with the size of (32,64,64,128); The maximum epochs(how many times each data point will be use) of the model setting is 200; The fraction of the validation set is 20% of the training data. According to the training loss curve , the training seems to stop around 80 epochs. 
  ![loss curve](/jonlangley2022.github.io/docs/assets/images/model/training_losscurve.png)
 To test the robustness of our MLP model , we try different initial value of weight and bias by changing the parameter of Random_state in scikit learn MLP regressor package:
 


### Applying Post-Prediction Inference
With the prediction model complete, we implemented the postpi functions and set out to see how much inference correction (if any) could be acheieved on a wide range of covariates of interest.

_insert permutation importance graph here

Postpi was applied to a total of 4 covariates, ranging from the top most important features to the prediction model (RushTD and QB Rating) to moderate/low importance features (TOP and 1stD)

-insert figure 2 here, dropdown for each of teh covariates

Two plots were made for each feature, one showing the covariate of interest compared to the _observed_ outcome and one showing the covariate of interest to the _predicted_ outcome.  

As can be seen for all of the features, the covariate compared to the predicted outcomes tend to show lower data spread and variance compared to the observed outcomes, increasing bias.

Next, we verify a key assumption of postpi, that the relationship between predicted and observed outcomes can be low-dimensionally modeled (such as via linear regression) despite the complexity of the prediction model

-insert figure 3 here, dropdown for each  of the covariates

For each of the covariates, a strong linear relationship is present between the observed and predicted outcomes for both the baseline linear regression model and the MLP Neural Network prediction model.

Inference correction can be conducted solely by using a linear or logistic regressioin relationship model fitted on the observed and predicted outcomes, but a more thorough and advanced correction can be achieved via the bootstrap method


### Results

### Limitations + Improvements



https://jonlangley2022.github.io

# 180B

# Applying Post Prediction Inference to the NFL: How We Strive to Reinvent Sports Analysis
 
 > By: Jonathan Langley, Sujeet Yeramareddy, and Yong Liu

### Introduction

For our 180B project, we decided to use our domain methodology of postpi (post prediction inference) and apply it to sports analysis based on NFL games.  We are designing a model that can predict the outcome of a football game, such as which team will win and what the margin of their victory will be, and then correcting the statistical inference for selected key features.  The main goals of our investigation is discerning which features most strongly determine the victor of a football game, and subsequently which features provide the most significant means of inferring the margin of that victory.  For example, does the home field advantage give a 50% higher chance of winning by 7 points?  Is the comparative offense to defense rating the most critical factor in securing a win?  Does or does not weather play a statistically significant part in influencing margin of victory?  These are just some of the questions we have brought up and seek to answer during the course of our research, and by conducting this project we are attempting to revolutionize the way NFL analytics are conducted via a more accurate statistical method of inference, postpi. 

### Data Collection
   Data source: stathead.com/football
   It is quite complicate and even more time-consuming to automated the scrapting process because we can only access to 100 rows of data at a time per feature (you had to click next at the bottom of the page to go on to rows 200-299, etc) and each feature will have different filter setting on the website.Therefore,we decided to manually copy and save the 10 csv files as we are interested in these  stats: fitst down, passing completion, rush yards,total yards, penalties, temperature at game day,first down. 
### Data Cleaning & EDA
 After merging all of the csv table we end up with a single csv with 5632 NFL games played from 2000 to 2021 but we actually have 11264 rows datapoints because of different home/away parameters. The combined dataset consisted of 64 total columns ranging from the year the game took place in, to the number of passes completed in the game, to percent of passes completed in the game. 
 A heatmap of null values allowed us to visually identify that 9 columns had 100-301 null values, with a tenth (temperature) having almost 2500 missing values.	


### Building Neural Network ML Model
 Through testing ,we finalized our N-N (MLPregressor) model with 4 hidden layer and with the size of (32,64,64,128); The maximum epochs(how many times each data point will be use) of the model setting is 200; The fraction of the validation set is 20% of the training data. According to the training loss curve , the training seems to stop around 80 epochs. 
 
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

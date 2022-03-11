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

  <img src="/assets/images/model/training_losscurve.png" alt="Alt text" title="Optional title">
 To test the robustness of our MLP model , we try different initial value of weight and bias by changing the parameter of Random_state in scikit learn MLP regressor package:
 


### Applying Post-Prediction Inference
The permutation graph shows the importance of each feature to the prediction model, but in order to accurately gauge the effectiveness of postpi on inference correction for our data we must apply postpi to each of our covariates, later on in our results we will feature our findings for two high importance features (RushTD and QBRating) and two moderate/low importance features (TOP and 1stD).
First, some definitions of what exactly postpi is.  Postpi is a method for improving statistical inference by correcting model bias and improving variance estimations.  Model bias refers to the difference between average prediction and the correct observation the model is attempting to predict, high bias is due to an under-fitted prediction model and leads to high training/testing error. Variance estimation relates to a model’s ability to predict on testing and unseen data, high variance is due to over-fitting on training data and leads to high test error.

Statistical inference (and the strength of the inference) refers to the analyst's capactiy to derive a trend of relationship between a covariate of interest and the outcome in a sample population, and then apply it to the entire population.

A hypothetical example to help understand this in the context of sports analytics is a "Home Field Advantage Inference" example.  A sports analyst takes a random sample of NFL games and finds statistically significant results that a team playing on its home turf has around a 10-15% higher chance of winning. Because of the analyst’s strong findings, they can provide reliable statistical inference across all NFL games (future or otherwise) that a team playing on their home turf has around a 10-15% higher chance of winning.

Now that we have some key definitions explained, we can move onto the actual implementation of postpi.  With our well tuned prediction model ready for use, we apply it on the test set to generate a list of predicted and observed outcomes.  A key aspect of post inference correction is producing a low dimensional (in our case linear regression) model to capture the relationship between the test set's observed and predicted outcomes.  We will later on use the relationship model to simulate "observed outcomes" by plugging in predicted outcomes and returning corrected predictions.  Inference correction by altering the prediction model would be possible for a simple model, however it would be nearly impossible to conduct meaningful improvement by altering a highly complex model such as MLP Neural Networks.  

Here we highlight our findings thusfar on our aforementioned four covariates of interest.  The first set of graphs compare the covariate of interest to the observed and to the predicted outcomes.  You'll see that for all 4, the predicted outcomes share a very similar relationship to the observed, but do tend to show less variance and more bunching near the central lines.

-insert dropdowns for the 4 figure 2 plots


In the next set of graphs we compare the relationship models generated on each covariate of interest from the NN and baseline (linear regression) models.  You can see that despite the complexity difference between neural network and linear regression, they all successfully generate a linear relationship between the observed and predicted outcomes.  Furthermore, it's clear to see that the MLP NN does a far better job across the board at capturing the relationship as the linearity is much stronger.


-insert dropdowns for the 4 figure 3 plots

Having established that the relationship models are strong and function as expected, the next step is implementing bootstrap based correction on the validation set.  The bootstrap repeatedly samples from validation data to better represent the entire population, generates an inference model for each iteration, and returns aggregated metrics for the inference models' corrected beta estimates, standard errors, test statistics, and p-values.

 






### Results

For every one of the featured covariates (and indeed for all of the covariates in our data set), the T-Statistics and P-Values graphs provide a clear answer to the to the question “to what degree, if any, will postpi provide statistical inference correction for our NFL sports analysis?” The answer to that being “not much”, apparently. While the exact reasoning as to why postpi didn’t help out much here can’t be confidently stated, we can speculate that the very high accuracy of our MLP Neural Network prediction model effectively expressed the link between each of our covariates and the game spread outcome. 




https://jonlangley2022.github.io

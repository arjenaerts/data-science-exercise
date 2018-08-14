# Questions
Please try to answer the following questions.
Reference your own code/Notebooks where relevant.

## Your solution
1. What machine learning techniques did you use for your solution? Why?
	Graphical probability model (white box): each customer is modeled as a single time-series and associated parameters are estimated using partial pooling (hierarchical Bayesian estimation). The main reason for using this approach is that it is much easier to incorporate knowledge about the structure of the data that you know beforehand (i.e. daily fluctuations, weekly effects, seasonal effects). Using a black-box method means that you essentially waste computational / time-resources to estimate a pattern that you know beforehand, assuming you have enough data to find the pattern in the first place. Another advantage of this method is that it is easy to use time-series for each customer that are of different length. Finally, using a probabilistic approach allows one to employ Bayesian methods, which faciliate hierarchical modeling and, more generally, regularization in a natural way. 
2. What is the error of your prediction?

   a. How did you estimate the error?
	I used mean absolute error over all observations   		
   b. How do the train and test errors compare?
    Due lack of time (see caveats below) I was not able to do validation. The train error (using the entire dataset) is 0.18. I'm quite sure this error can be reduced substantially by including periodic effects.
   c. How will this error change for predicting a further week/month/year into the future?
    The test error is likely to increase, as the world changes over time and more data comes in, potentially changing your parameter estimates
3. What improvements to your approach would you pursue next? Why?
 Due to lack of time, I was not able to implement two less advanced models (see caveats), namely one with dummies and one with multiple states. It is clear from the data exploration (see notebook) that there are strong periodic effects on the yearly level, weekly level and daily level. To deal with the first two one can incorporate dummies into the model, by incorporating dummy variables into the regression equation for the lambda parameter for each customer separately; the dummy coefficients would be customer specific and can also be estimated using a hierarchical framework. Daily variation can be dealt with by identifying two or more states and letting the customer transition between the states via a Markov chain; to make the states observable one can manually set a threshold for energy usage.  
 
 There are three additional improvements that can be combined. First, the transition probability between being in the low-usage and high-usage state should be dependent on the particular time of the day, such that for each customer and for each day the model can predict accurately when the customer is using little energy. Second, in the current framework the two states where artificially constructed by putting a threshold. Instead of explicitly creating these states one can introduce hidden states, which makes the model more general (and will increase the computational burden significantly). Third, a sensitivity analysis for the prior distributions should be performed. 
4. Will your approach work for a new household with little/no half-hourly data?
   How would you approach forecasting for a new household?
    Yes, because the global parameters that are estimated with the hierarchical algorithm can be used to randomly sample the local parameters of a model for a new customer. The new customer is thus viewed as a typical customer, which is as good as it gets, unless we have data on this customer (e.g. age, sex, socio-economic background) that we can incorporate in the model as well. 

# Caveats
This week I am moving continents, so I have been quite busy and therefore had less time than normal. Moreover, beause I had to hand in my laptop at work, I had to use my fiance's MacBook, leading to many technical problems. Even installing VirtualBox to set up a Ubuntu VM was problematic. Then there were severe working memory problems with the VM because Macs have large overhead, so I could not use the PyCharm IDE and other things. All in all I spent more than a full day setting up my data science environment. Therefore, I was not able to do validation testing and run two more models (plus some benchmarks). My apologies for this; if required I can include this next week when I am back in Netherlands.
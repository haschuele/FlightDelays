# Predicting Flight Delays

The overall objective of this project was to predict departure delays greater than 15 minutes, 2 hours before takeoff. The business case is described in more detail in the [final report](https://github.com/haschuele/FlightDelays/blob/main/Final%20Report.pdf), but the results of this analysis have significant benefits for both airlines and passengers.

In Phase I of this project, we sorted out project management and our computing environment. We also made plans for our data pipeline and did our initial EDA to identify variables of interest.

In Phase II of this project, we first performed a join to combine our datasets. This was followed by the iterative data cleaning, EDA, and feature engineering process. We ran a logistic regression model as our baseline, after which we conducted LASSO regularizaiton. The resulting 24 features were used in another logistic regression, which showed substantial improvements over our baseline.

In the final Phase III, we started by performing additional feature engineering and EDA to ensure we were getting the most utility out of the data. We then experimented with a variety of models (logistic regression, random forest, gradient boosted decision trees, multilayer perceptron, and long short-term memory), hyperparameters, and LASSO regularization to see how they performed on the validation data. The best model was selected, trained on the full data, and tested on an unseen data cut to give us our final performance.

Our final model showed a 35% increase in F1 score from our baseline model. Further investigation showed that precision and recall were very similar, indicating we have a balanced model. Code snippets for this project can be found [here](https://github.com/haschuele/FlightDelays/blob/main/Code%20Snippets.md).

This group project was conducted in the UC Berkeley MIDS' "Machine Learning at Scale" course.

# Mun_Flight_Fare_prediction
Flight Prices are dynamic in nature i.e they change from time to time . 
so in this project we are building two  models that will predict the Flight Fares and we are going to compare which Model is going to give accurate results.

Learning Outcomes from the Project:

1.Exploratory Data Analysis
2. Understanding, Processing, and visualizing the data 
3.Making the machine learning models
4. Model comparison and choosing the best model for accurate results.
 
we have two datasets one for training and other for testing the results in this project we use Lasso and Decision tree regression models and find out which is the best model to predict the flight prices for the given data.

The categories that our dataset contains are Airline, Date_of_Journey, Source, Destination, Route, Arrival_Time, Duration, Total_Stops, Additional_Info, Price

we have loaded the given data into data frames with the help of pandas and have performed many operations on them such as Data pre-processing, Data Categorization, Encoding 

Once the data is processed then we have used that data for modelling  we have used Lasso and Decision tree regression models as part of my project in the end we can conclude which model gives us the more accurate results.

We trained both the Lasso regressor and Decision tree regression models and we came to the conclusion that Decision tree regression model has achieved a r2-score of 82% whereas the lasso regression model has only got 42% so with the help of Decision tree regression we can predict the flight fares that were actually close to the original prices.

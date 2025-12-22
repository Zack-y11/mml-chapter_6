import numpy as np

#Mockup data for house prices
#row: different houses
#Columns are differente features of the houses [area, bedrooms, bathrooms]
house_matrix = np.array([
    [1500, 3, 2],
    [2000, 4, 3],
    [800, 1, 1]
])

#The model (The Weights Vector)
#Values for [sq ft, bedrooms, bathrooms]
weight_vector = np.array([100,50000, 30000])

#The operation to predict the prices of the houses
#This multiplies the house matrix by the weight vector to get a result for each house
predict_prices = np.dot(house_matrix, weight_vector)

#Print the results
print("House Matrix: ", house_matrix)
print("Weight Vector: ", weight_vector)
print("Predicted Prices House A: ", predict_prices[0])
print("Predicted Prices House B: ", predict_prices[1])
print("Predicted Prices House C: ", predict_prices[2])

# Predictive analysis/modelling of Eating out problem

The project involves the utilization of a real-world dataset, "zomato_df_final_data.csv," containing comprehensive information about various restaurants located in the suburbs of Sydney. The primary objective of this analysis is to perform exploratory data analysis, feature engineering and develop regression and classification models to predict restaurants' customer ratings.  
# Data Description  
Contains more than 10,000 records of restaurants' in Sydney in the year 2018. 
| Column          | Description                                                  | Example                                      |
|-----------------|--------------------------------------------------------------|----------------------------------------------|
| 'address'       | Restaurant's address (text)                                  | 371A Pitt Street, CBD, Sydney                |
| 'cost'          | Average cost for two people in AUD (numeric)                | 50.0                                         |
| 'cuisine'       | Cuisines served by the restaurant (list)                    | [Thai, Salad]                                |
| 'lat'           | Latitude (numeric)                                          | -33.876059                                   |
| 'link'          | URL (text)                                                  | [https://www.zomato.com/sydney/sydney-madang-cbd](https://www.zomato.com/sydney/sydney-madang-cbd) |
| 'lng'           | Longitude (numeric)                                         | 151.207605                                   |
| 'phone'         | Phone number (numeric)                                      | 02 8318 0406                                |
| 'rating_number' | Restaurant rating (numeric)                                 | 4.0                                          |
| 'rating_text'   | Restaurant rating (text)                                    | Very Good                                    |
| 'subzone'       | Suburb in which the restaurant resides (text)               | CBD                                          |
| 'title'         | Restaurant's name (text)                                    | Sydney Madang                                |
| 'type'          | Business type (list)                                        | [Casual Dining]                              |
| 'votes'         | Number of users who provided the rating (numeric)           | 1311.0                                       |
| 'groupon'       | Is the restaurant promoting itself on Groupon.com? (boolean) | False                                        |




## Authors

- [@Adnan525](https://github.com/Adnan525)


## Features

* Part A (dsts_a1) has the exploratory analysis of the categorical and numerical variables.
     * In this section, an analysis is conducted to determine the quantity of culinary offerings available within restaurants located in Sydney.
    - The section also contains a pricing/cost analysis to support the statement that "Restaurants with excellent rating are mostly very expensive while those with poor rating are rarely expensive"
    - Lastly, number of visalisation aids like histogram, bar chart, box plot were used to conduct exploratory anaylis of the variables of the dataset.
- Part B (dsts_geopandas) focuses on data visualization, with the primary objective being the representation of restaurant concentration within a specific suburb, considering a particular culinary theme
    - In the context of Part B of the analysis, the process involved the utilization of a geojson file named 'sydney.geojson,' which was included within the dataset. This file was employed for the purpose of merging with the original dataset, thereby introducing additional attributes, specifically the SSC_CODE (State Suburbs Code), and incorporating geometric polygonal information.
    - Matplotlib was employed to generate a heatmap illustrating the distribution of restaurants within a particular suburb, with the representation of data being contingent on cuisine type.
    ![Matplotlib heatmap](https://github.com/Adnan525/syd_restaurant/blob/master/matplotlib_heatmap.JPG)
    - Folium was used to generate the same heatmap but had interactive features, like zoom, moving capability etc.  
    ![Folium heatmap](https://github.com/Adnan525/syd_restaurant/blob/master/folium_heatmap.JPG)

- Part C contains 2 regression models that predicts the rating of a restaurant, based on features available in the dataset. The section also contains a binary-classification model using logistic regression that classifies the restaurants between (poor, average) = 1 and (good, very good, excellent) = 2.
    - Regression model 1 had - MSE : 0.1469729505451935 and RMSE : 0.2540572715441253
    - Regression model 2 used gradient descent and standardized values, had - MSE : 0.1490877358715363 and RMSE : 0.2433239445572284
    - The classification model 3 had -

| Metric           | Value                  |
|------------------|------------------------|
| Accuracy         | 0.9853862212943633     |
| Precision        | 0.9863157894736843     |
| Recall           | 0.9915343915343915     |
| F1 Score         | 0.9889182058047493     |
| ROC AUC Score    | 0.9825558136533745     |
| Confusion Matrix|937   8                
|                 |  13  479               |

- Part 3 alos contains 4 classification models using Support vector Machine with Linear and Radial Bias Function Kernel, Decision Tree and Random Forest modelling. The accuracy metrics for those models are -
  
| Model          | Accuracy | Precision | Recall  | F1 Score | ROC AUC  |
|----------------|----------|-----------|---------|----------|----------|
| Linear SVM     | 1.000    | 0.986316  | 0.991534| 0.988918 | 0.982556 |
| RBF SVM        | 0.856646 | 0.986316  | 0.991534| 0.988918 | 0.982556 |
| Decision Tree  | 1.000    | 0.986316  | 0.991534| 0.988918 | 0.982556 |
| Random Forest  | 1.000    | 0.986316  | 0.991534| 0.988918 | 0.982556 |


## Installation

Requirements :
- Python 3.8 or above
- Install all the required packages - pandas, numpy, geopandas, sci-kitlearn, seaborn, matplotlib, folium
- Jupyter notebook or similar notebook environments to run the ".ipynb" files
- The modelling tasks are available in a docker container at - https://hub.docker.com/r/ghost525/dsts_a1
- Provided that docker is installed, run the following command - 

```bash
docker pull ghost525/dsts_a1
```
    
## Usage
Running the ipynb files cell by cell will guide a user through the analysis and model buildin process. Also the user would see data cleaning and feature engineering process.  
Running the docker container will produce the accuracy metrics for the regression models and the logistic regression binary-classification model. Expected output -  
```bash
=======================
model_regression_1
Mean Squared Error: 0.1469846169099422
R-squared: 0.25399806037690364
=======================

=======================
model_regression_2
Mean Squared Error: 0.14910993202209347
R-squared: 0.24321129078626968
=======================

=======================
Accuracy: 0.9784272790535838
Precision: 0.9780334728033473
Recall: 0.9894179894179894
F1 Score: 0.9836927932667017
ROC AUC Score: 0.9733675312943605
Confusion Matrix:
 [[935  10]
 [ 21 471]]
=======================
```
Also the dsts_geopandas has the following functions- 
```python
# matplotlib plot
show_cuisine_densitymap(cu: str, result: geopandas.geodataframe.GeoDataFrame, gdf: geopandas.geodataframe.GeoDataFrame)
"""
    cu : string vale, name of the cuisine for the heatmap
    result : merged and filtered dataframe by cu, must contain geometry details such as polygon
    gdf : base geopandas dataframe, used as the base of the heatmap, filtered suburbs are plotted on top of this
"""

# inteactive folium
display_map(cu: str)
"""
    cu : string vale, name of the cuisine for the heatmap
"""
```
The function calls will directly result the heatmaps.



## Acknowledgements

 - [Geeks for geeks](https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/)
 - [Scikit Learn](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
 - [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/06/guide-geospatial-analysis-folium-python/)
- [Real Python](https://realpython.com/linear-regression-in-python/)
- [Awesome README](https://github.com/matiassingers/awesome-readme)

# Tableau Dashboard for the dataset :  
[Tableau](https://public.tableau.com/app/profile/muntasir.adnan/viz/adnan525/Dashboard1)

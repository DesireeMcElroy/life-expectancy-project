<div style="text-align:center"><img src="https://user-images.githubusercontent.com/69991789/125513616-a74f07d9-849b-49b8-9960-7436c7cbf4bb.png" /></div>

What country characteristics drive the average life expectancy of it's citizens? Is the quality of life lower in the underdeveloped countries than those that are developed? Do vaccination rates such as hepatitis b and polio play a part in infant mortality death rates? Is a country's total expenditure correlated to the BMI rate of their population? How can we predict average life expectancy of a population accurately? Can pinpointing drivers help improve life expectancy in certain countries?\
In my project, I explore each of these inquiries utilizing the World Health Organization's Life Expectancy dataframe. Through visualizations and statistical testing, I am able to pinpoint which data characteristics possess positive and negative correlations to a country's average life expectancy. Utilizing supervised machine learning via regression, I was able to create a machine learning model that accurately predicted life expectancy with an average variance two to three years based off of the dataset's top driving features.


## Project Goals
1. Wrangle WHO life expectancy dataset by acquiring and cleaning the data to increase usability for project.
2. Utilize wrangled dataframe to explore correlating features to life expectancy as well as relationships between independent variables via data visualizations, statistical testing and clustering methods.
3. Create machine learning models to predict life expectancy and compare end result to baseline model.

## Replicate My Project
### tools/libraries:
    1. python
    2. pandas
    3. scipy
    4. sci-kit learn
    5. numpy
    6. matplotlib.pyplot
    7. seaborn
* Steps to recreate
    1. Clone this repository
    - https://github.com/DesireeMcElroy/life-expectancy-project
    2. Import WHO Life Expectancy dataset: Employ data science pipeline to wrangle, explore and model predictions.

## Key Findings
1. Through visual exploration and statistical testing, I was able to support my suspicion of driving characteristics of average life expectancy.
2. Using the recursive feature method, I confirmed which features would be used for my final model.
2. I first created a baseline model using the mean average of life expectancy across the board.I initially obtained an RMSE score of 9.2. This score was pretty high considering life expectancy was off by plus or minus nine years.
3. From there I created five competitive models using multiple regression algorithms. Three models outperformed my baseline with over 50% increased accuracy.
4. In the end I moved forward with Model 5 (PolynomialRegression algorithm with a degree of 3) to test on the unseen test data set. The results were as follows:
    - train dataset: RMSE = 2.72
    - validate dataset (unseen, used to ensure no overfitting) dataset: RMSE = 3.27
    - test dataset (unseen, used to verify prediction accuracy) dataset: RMSE = 3.43
5. Despite a small chance of the model being overfit, Model 5 performed well all around on unseen data and accurately predicted life expectancy within two to three years.

## Drawing Board
View my trello board [here](https://trello.com/b/OUlKpE5E/life-expectancy-project).

------------

I want to examine these possibilities:
1. Does the year of the country's data have any correlation to life expectancy?
2. Do vaccination rates (hep_b, measles) have a positive correlation to life expectancy?
3. Are there possible clusters among the independent variables?
4. Will robust scaling work better considering the dataframes outliers?


-------

I will verify my hypotheses using statistical testing and data visualizations. By the end of exploration, I will have identified which features are the best for my model.

During the modeling phase I will establish a baseline model based off the distribution of life expectancy and then use my selected features to generate multiple competitive regression models. I will evaluate each model using the highest driving features of life expectancy and compare each model's performance to the baseline. Once I have selected the best model, I will subject it to the unseen test data sample and evaluate the results.


## Data Dictionary

#### Target
Name | Description | Type
:---: | :---: | :---:
life_expectancy | The average life expectancy in years of that country's population | float
#### Features
Name | Description | Type
:---: | :---: | :---:
country | The name of the specified country | object
year | The year the data of the observation was recorded | int
status | Indicates whether a country is developed or still developing | int
adult_mortality | Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population) | float
infant_deaths | Number of Infant Deaths per 1000 population | int
alcohol | Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)  | float
pct_expenditure | Expenditure on health as a percentage of Gross Domestic Product per capita(%) | float
hep_b | Hepatitis B (HepB) immunization coverage among 1-year-olds (%) | float
measles | Measles - number of reported cases per 1000 population | int
bmi | Average Body Mass Index of entire population | float
under_five_deaths | Number of under-five deaths per 1000 population | int
polio | Polio (Pol3) immunization coverage among 1-year-olds (%) | float
total_expenditure | General government expenditure on health as a percentage of total government expenditure (%) | float
diptheria | Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%) | float
hiv/aids | Deaths per 1 000 live births HIV/AIDS (0-4 years) | float
gdp | Gross Domestic Product per capita (in USD) | float
population | Population of the country | float
thinness_10-19yrs | Prevalence of thinness among children and adolescents for Age 10 to 19 (% ) | float
thinness_5-9yrs | Prevalence of thinness among children for Age 5 to 9(%) | float
income_comp_resources | Human Development Index in terms of income composition of resources (index ranging from 0 to 1) | float
yrs_education | Number of years of Schooling(years) | float




## Recommendations/With More Time:
1. I would prefer to continue to impute the remainder of my nulls as opposed to just dropping the rest of them.
2. I would explore more clustering possibilities to create new features and see how I can create more correlated features.
3. I would assess the outliers and/or utilize different scaling methods to see if that has a varying impact on model performance.


Resources:

Find the dataset I used [here](https://www.kaggle.com/kumarajarshi/life-expectancy-who).


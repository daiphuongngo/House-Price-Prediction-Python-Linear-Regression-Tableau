# Project: **Analyzing Housing Affordability: Exploring Key Influential Features by Regression and Statistical Modeling**

![Harvard_University_logo svg](https://github.com/user-attachments/assets/02c5241a-3095-4593-8aea-cfd65ddf4cce)

![Harvard-Extension-School](https://github.com/user-attachments/assets/95a203c1-fd4e-4881-94d4-0664f989f814)

## **Master of Liberal Arts, Data Science**

## **CSCI E-83 Fundamentals of Data Science in Python**

## Professor's name: **Stephen Elston**

## Author's name: **Dai Phuong Ngo (Liam)**

## **Introduction and Background**

The housing market has long been a crucial area of interests for businesses, individuals and especially policy makers. In the mortgage banking industry where I am working for at the moment, understanding factors impacting on housing affordability is critical for tailoring financial products, advising clients and managing risks from different perspectives. In Canada particularly, housing affordability has become increasingly challenging due to rising sales prices, limited inventory because of slow construction versus high demand, and economic uncertainty. This triggered the requirement for mortgage providers and underwriters, real estate brokers, consumers to evaluate the most influential factors driving housing affordability.

The issues might face complexity due to the wide range of direct features impacting on sales price, such as living space, condition, property type or less obvious features, such as, year remodeled or basement size. An unwanted fact is that the distribution of sales price is often skewed with extreme outlers complicating further analysis. Traditional evalutions might not succeed in understanding all of these variations, therefore, empowering advanced statistical techniques is important to understand and diagnose affordability better.


## **Project Goal**

This project aims to analyze the key housing features that affect sales prices and housing affordability to provide actionable insights. Inferences retrieved from the data can help to identify features having greatest impact on affordability and generate further insights for homebuyers, who have to make informed decision on which housing option is compatible with their affordability, real estate brokers, who have to customize advice and recommendations to individual needs, and mortgage providers or banks, like my current company, who have to set up financing solutions based on metrics of housing affordability. Therefore, the final goal of this project is to develop statistical solutions with sampling techniques, OLS regression models and Bayesian analysis to assess contributions from the most influential housing features to housing affordability. This emphasis will be on inference, rather than prediction, and will address upcoming challenges expressed by skew distributions in prices and their outliers to make sure conclusions are drawn in a statistically reliable, practical and relevant way.


## **What This Project Is Not About**

It is import to clarify that this project is not about price prediction or prediction's optimization. Instead, it concentrates on statistical inference, which determines the impact of different features on affordability and provides a statistical framework to interpret their impacts to indentify the most crucial influencers. Machine Learning algorithms such as tree-nased models or neural networks are not applicable in this project as it focus on capturing relationships among features. Also, this project does not set certain affordability thresholds manually and affordability's classification.

## **Data**

A challenge arises with the data availability for the Canadian context that I am unable to find a good enough or ready-to-consume data with domain in Canadian housing affordability. I tried to look through websites of Government of Canada or Canada Statistics but their data are scattered, not yet combined, time series type and require significant amount of time to find enough data aspects for manipulationa and combination, regarless of their individual dataset's completeness across features. Therefore, I decided to focus on statiscal modelling techniques analyzing housing affordability with usage of this data from the American housing context of Ames city.

The data comes directly from Kaggle with two sets: training and testing. I will use the training for data exploration, preparation, assessment and model development before applying the best model on the testing set. The data has 79 explanatory variables which are manageable and describe aspects of residential properties in Ames, Iowa. This data provides an excellent source to study housing trends and affordability with a good amount of meaningful features and data records describing different aspects of housing. The modelling will take all features as influencers to be analyzed, visualized, sampled and modelled to assess the final set of best features on different samples of the US house prices and affordability in Ames. Later, the training data can be split into 2 samples: training and testing for further model evaluation.

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

(TO BE CONTINUED, please refer to this final notebook in this repository for analysis and codes: https://colab.research.google.com/drive/1gNkXxY6OfcLWn5zHZhHML8VBmNIWeiKl?usp=sharing )

## Languages, Machine Learning, Deep Learning models, Tools:

- Python

- Tableau

- Linear Regression

- GridSearchCV

- Deep Neural Network

Platform: Spyder (Anaconda)

Statistics model: Linear Regression

## Modeling & Evaluation:

The model evaluated on Root Mean Square Error (rsme) receives a result at roughly $38,700 which shows that this error in predicting house prices is satisfactory. The variable having the most impact on Sale Price is Overall Quality which indicates the highest correlation index at 0.79 among all features.
Homebuyers in the US should take care of the Overall Quality of their property as their main feature to consider before selling and buying. What comes next is when to do it as before the Financial Crisis 2007-2008, house prices might be higher with more flexible sale types and conditions. 
As the Crisis began in 2007, a drop in prices could be a great loss for sellers but a once-in-a-lifetime deal for buyers if they afforded to purchase as previous bank loans became tightened until further actions aiding the US economy proceeded by the US President and US Congress.


## Table of Content
### I. Overview	

79 explanatory variables are provided from the dataset and describe aspects of residential properties in Ames, Iowa. This evaluation model will take 10 features (listed below) as the main influencers to be analyzed and visualized with Python and basic Linear Regression to assess the final Root Mean Squared Error of the US house prices in Ames. With further emphasis on the period of 2006-2010, when the Financial Crisis 2007-2008 happened, this model will discuss which variable becomes the most influential feature affecting the Sale Price in Ames.

10 key features for the analysis: 'SalePrice', 'MSSubClass', 'MSZoning', 'LotArea', 'BldgType', 'HouseStyle', 'YearBuilt', 'YrSold', 'SaleType', 'SaleCondition', ‘OverallQual

### II. Preprocessing	

General data: 1454 x 54
+ Train: 1018 x 54
+ Test: 436 x 54

Sale Prices mainly focus on a range from US$ 100,000 up to 200,000 through the 6-year period. So, this chart shows an initial glance of a price range with high spendings in the housing market. 

![image](https://user-images.githubusercontent.com/70437668/138621471-c4ff09c8-3e72-453b-933d-d89c1b67498e.png)

#### 1/ MsSubClass: Identifies the type of dwelling involded in the sale	

![image](https://user-images.githubusercontent.com/70437668/138621486-a027a010-ec84-4e12-a7b1-c2038448a186.png)

Generally, the identifies the type of dwelling involved in the sale are assumed to have strong positive correlation with Sale Prices. However, Sale Prices are randomly distributed in all types of dwelling involved in the sale instead. For most, 1-STORY 1946 & NEWER ALL STYLES and 2-STORY 1946 & NEWER takes the most proportion of identities, mainly ranging from around US$20,000 to slightly over 400,000 per property for 1-STORY 1946 & NEWER ALL STYLES. While it takes a span from around US$120,0000 to 460,000 per property for 2-STORY 1946 & NEWER. This proves that homebuyers processes home deals with any kinds of properties, especially 1-story and 2-story houses. We can see that the weak negative correlation between MSSubClass vs SalePrice reaches only at -0.084. 

#### 2/ MSZoning: Identifies the general zoning classification of the sale

![image](https://user-images.githubusercontent.com/70437668/138621526-fcbbc0cc-b263-43ac-8086-054bff0533f4.png)

As a whole, homebuyers aimed at a property rate starting from around US$100,000 per property in all zones except some areas such as Agriculture, Commercial, Industrial and Residential Low Density Park with q25 beginning from US$50,000. While their q75 is able to reach at nearly US$ 100,000. To be more specific, Floating Village Residential takes the greatest Sale Prices with q25 arring at nearly US175,000 and q75 extending up to US$250,000. Meanwhile, Residential Low Density takes the second place with q25 almost extending to US$150,000 and q75 getting as fas as approximately US$225,000. All in all, most house deals distribute at and above US$100,000 per property proves that house values are considered to be at high rates for a majority of homebuyers.

#### 3/ LotArea: Lot size in square feet	

![image](https://user-images.githubusercontent.com/70437668/138621568-d92a43d8-483c-4b50-b166-c5bf7d12b526.png)

As we can see intuitively, a great number of house merchants takes place in properties below 50,000 square feet per property. These sizes cover up a range of prices largely below US$400,000. It is supposed that the larger the lot area, the greater the price. However, this is proved uncertain as the Lot Area has a small positive correlation with Sale Price at an index of 0.264. This means that Lot Area would not really affect the overall Sale Price even if the lot size could expand further. So, this also means that in the housing market, Lot Area can be bigger but it has a minor impact on any house values.

![image](https://user-images.githubusercontent.com/70437668/138621613-efaf37f5-f3a8-4c94-84c8-71c823285137.png)

#### 4/ BldgType: Type of dwelling	

![image](https://user-images.githubusercontent.com/70437668/138621620-796738f1-895a-4a43-babf-0bb937976521.png)

Another proof to confirm that Sale Prices all start from US$100,000 and above in all building types is the chart of Building Type vs Sale Price. It is highlighted that the Single-family Detached building takes the longest range of Sale Price with q25 above US$130,000 and q75 very close to US$ 225,000. Followingly, Townhouse Inside Unit and Townhouse End Unit come at the next places. These are the most common properties for single families with a decent lot area below 50,000 square feet. It can be seen that the more separated between a family and another for a house, the higher the price. So, homebuyers tend to buy more single units than a combined or shared property with other families. High mortgage approval rates led to a large pool of homebuyers, which drove up housing prices. This appreciation in value led large numbers of homeowners (subprime or not) to borrow against their homes as an apparent windfall. This "bubble" would be burst by a rising single-family residential mortgages delinquency rate beginning in August 2006 and peaking in the first quarter, 2010.

#### 5/ HouseStyle: Style of dwelling	

![image](https://user-images.githubusercontent.com/70437668/138621645-0b9a5d63-1337-40f6-8e7a-5fdb09c41bc7.png)

This chart also illustrates that all house styles have a range of prices mostly starting from US$100,000 and above no matter what statuses of they are or how many stories they have. We can easily see that more stories in a single unit and / or finished status of the house can lead to higher prices for homebuyers. Specifically, 2-story house takes the longest range and it is also the most expensive style when its q25 begins above US$150,000 and q75 ends up nearly at US$ 250,000. Another thing to mention is that Two and one-half story: 2nd level finished and One story (finished) take high prices as well with their q75 in the mid US$210,000-230,000. In the meantime, unfinished houses such as the one and one-half story: 2nd level unfinished is the least preferred with both q275 and q75 below US$ 120,000.

#### 6/ YearBuilt: Original construction date	

![image](https://user-images.githubusercontent.com/70437668/138621663-c1b9d2f9-5e56-4b30-8bef-a28d82858b01.png)

![image](https://user-images.githubusercontent.com/70437668/138621665-d5ba9bc4-05e5-4f24-9193-a8c8382c377b.png)

According to the lm chart, it can be seen that there is a moderate positive correlation between Year Built and Sale Price from 1872 to 2010 at 0.523. Some highlights can be mentioned that there were several long-term downturns in newly built houses throughout The World War I (1914-1918), The Great Depression (1929-1933) and The World War 2 (1939-1945). In contrast, during the Economic Crisis (2007-2008), houses built in these years and in the previous preriod have a surge in Sale Price as demand rises up with more lax mortgages in the previous crisis. 

#### 7/ Yrsold: Year Sold	

![image](https://user-images.githubusercontent.com/70437668/138621684-a59f7221-1d06-4833-8af6-6f4752fe34f5.png)

Logically, SalePrice would increase due to inflation annually. However, from 2006 to 2008, housing prices started to decrease, showing in q50, q75, q99 and range from q25 to q75 when comparing with 2007. The reason was that the 2007-2008 economic crisis had affected the housing market. There was also a slightly negative correlation between Year Sold and Sale Price at -0.029. This minor index was probably thanks to the Sale Price then started to recover at a slow velocity in 2009 and 2010 after United States President Barack Obama and US Congress's multiple regulatory and long-term responses. Therefore, there would be high prices generally for homebuyers to buy with newly revised mortgages.

Regulatory proposals and long-term responses Further information: Obama financial regulatory reform plan of 2009, Regulatory responses to the subprime crisis, and Subprime mortgage crisis solutions debate United States President Barack Obama and key advisers introduced a series of regulatory proposals in June 2009. The proposals address consumer protection, executive pay, bank financial cushions or capital requirements, expanded regulation of the shadow banking system and derivatives, and enhanced authority for the Federal Reserve to safely wind-down systemically important institutions, among others. In January 2010, Obama proposed additional regulations limiting the ability of banks to engage in proprietary trading. The proposals were dubbed "The Volcker Rule", in recognition of Paul Volcker, who has publicly argued for the proposed changes. The US Senate passed a reform bill in May 2010, following the House, which passed a bill in December 2009. These bills must now be reconciled. The New York Times provided a comparative summary of the features of the two bills, which address to varying extent the principles enumerated by the Obama administration. For instance, the Volcker Rule against proprietary trading is not part of the legislation, though in the Senate bill regulators have the discretion but not the obligation to prohibit these trades. 

#### 8/ Sale Type	

![image](https://user-images.githubusercontent.com/70437668/138621727-fc603799-d676-47d5-9969-eb7ae8fd2508.png)

According to the boxplot chart, overall, all types of sales are over US$100,000 per property. This is at a high rate for the housing market. It seemed that homebuyers tended not to buy properties at a moderate rate but spend a large amount of banking loans thanks to mortgages on housing matters. Furthermore, they seemed to buy new ones at the highest range of prices. When looking at the swarm plot chart, Warranty Deed - Conventional's payment method took the greatest amount of housing sales and Home just constructed and sold took the second place. This is because there were too many homeowners with questionable credit and that banks had allowed people to take out loans for 100% or more of the value of their new homes.

### 9/ Sale Condition	

![image](https://user-images.githubusercontent.com/70437668/138621741-e4a100c0-d6ef-4001-9175-fe087faaefa0.png)

According to these 2 charts, housing sale cases mainly forms in groups of Normal Sale, Abnormal Sale -  trade, foreclosure, short sale and Partial	 Sale -Home was not completed when last assessed (associated with New Homes). Specifically, Normal Sales has the most cases ranging strongly from US$ 100,000 up to 300,000 while the range of the Partial Sales fluctuates shorter from US$ 150,000 to 250,000 and the one of Abnormal Sale spans the shortest from US$100,000 to 150,000. In general, these 3 conditions all started from around US$100,000 and above which illustrates house deals are completed at a high rate level. This also draws attention to an overview of different conditions for the majority of housing deals with normal and short processes.

The precipitating factor for the Financial Crisis of 2007–2008 was a high default rate in the United States subprime home mortgage sector, i.e. the bursting of the "subprime bubble". This happened when many housing mortgage debtors failed to make their regular payments, leading to a high rate of foreclosures. While the causes of the bubble are disputed, some or all of the following factors must have contributed. Also, low interest rates encouraged mortgage lending.

Regarding the Securitization, any mortgages were bundled together and formed into new financial instruments called mortgage-backed securities, in a process known as securitization. These bundles could be sold as (ostensibly) low-risk securities partly because they were often backed by credit default swaps insurance. Because mortgage lenders could pass these mortgages (and the associated risks) on in this way, they could and did adopt loose underwriting criteria (due in part to outdated and lax regulation).
Mortgage guarantees. Many of the subprime (high risk) loans were bundled and sold, finally accruing to the quasi-government agencies Fannie Mae and Freddie Mac. The implicit guarantee by the US federal government created a moral hazard and contributed to a glut of risky lending.

![image](https://user-images.githubusercontent.com/70437668/138621773-f5cb47fd-71dd-49d1-a8ac-e0c426644b65.png)

#### 10/ OverallQual: Rates the overall house material and completion 

![image](https://user-images.githubusercontent.com/70437668/138621779-8fba90a9-885b-4c7d-aa7a-008c302268d4.png)

The Overall Quality has a very strong correlation with Sale Price as its correlation index is roughly 0.79. It seems that they have a linear relationship. In general, the higher point a property could get, the better prices its value could reach. This trend was not changed even through the assessed period when the Financial Crisis 2007-2008 took place.

### III. Model & Evaluation	

The model evaluated on Root Mean Square Error (rsme) receives a result at roughly $38,700 which shows that this error in predicting house prices is satisfactory. The variable having the most impact on Sale Price is Year Sold which indicates the highest correlation index at 0.79 among all features. Home buyers in the US should take care of the Overall Quality of their property as their main feature to consider before selling and buying. What comes next is when to do it as before the Financial Crisis 2007-2008, house prices might be higher with more flexible sale types and conditions. In fact, as the Crisis began in 2007, a drop in prices could be a great loss for sellers but a once-in-a-lifetime deal for buyers if they afforded to purchase as previous bank loans became tightened until the US President Obama pushed the US Congress for further actions aiding the US economy.

### IV. Implement Deep Neural Network

### Goal

* Use methods of scaling, optimizer, learning)rate, model architecture
* **Apply GridSearchCV in Koeras to find the best optimizer, learning_rate, momentum**
* **After training the model, I will evaluate the performance on Test set**

```
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten
# Regression: use KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
```

```
# ===========================================
# Case 1: learning_rate=0.01, momentum=0.9
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

def create_model(optimizer='sgd', learning_rate=0.01, momentum=0.9):
  model = Sequential()
  model.add(Dense(128, activation='relu', input_shape=X_train.shape[1:]))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='linear'))
  # model.add(Dense(8, activation='relu', input_shape=X_train.shape[1:]))
  # model.add(Dense(1, activation='linear'))
  my_optim=None
  if optimizer == 'adam':
    # if using optimizer as adam, there will be no learning_rate
    my_optim = Adam(learning_rate=learning_rate)
  elif optimizer == 'sgd':
    my_optim = SGD(learning_rate=learning_rate, momentum=momentum)
  elif optimizer == 'rmsprop':
    my_optim = RMSprop(learning_rate=learning_rate, momentum=momentum)
  
  model.compile(loss='mse', optimizer=my_optim, metrics=['mae',RootMeanSquaredError()])
  return model
```

### Define the grid search parameters
```
optimizer_values = ['adam', 'sgd', 'rmsprop']
lr_values = [0.01, 0.1]
momentum_values = [0.0, 0.9]

param_grid = {
    'optimizer': optimizer_values,
    'learning_rate': lr_values,
    'momentum': momentum_values
}
```

```
model = KerasRegressor(build_fn=create_model, epochs=80, verbose=1)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train_scaled, y_train_scaled)
print("Best model using: %s" % (grid_result.best_params_))
```

```
Epoch 80/80
37/37 [==============================] - 0s 1ms/step - loss: 0.0816 - mae: 0.1982 - root_mean_squared_error: 0.2857
Best model using: {'learning_rate': 0.01, 'momentum': 0.0, 'optimizer': 'sgd'}
```

```
best_model = grid_result.best_estimator_
best_model.model.evaluate(X_test_scaled, y_test_scaled)
```

```
10/10 [==============================] - 0s 1ms/step - loss: 0.1051 - mae: 0.2344 - root_mean_squared_error: 0.3241
[0.1050662025809288, 0.23435331881046295, 0.32413917779922485]
```

### Compare MAE of y_test_scaled and y_pred_inverse

Metrics are to evaluate the model performance but how much metrics should be depends on which I would like to define

```
y_pred = best_model.predict(X_test_scaled)
y_pred_inverse = y_scale.inverse_transform(y_pred) # after scaling, I have to inverse_transform to see the real house price

from tensorflow.keras.metrics import MeanAbsoluteError
mae = MeanAbsoluteError()
print(mae(y_test_scaled, y_pred_inverse))
```

```
10/10 [==============================] - 0s 1ms/step
tf.Tensor(186325.62, shape=(), dtype=float32)
```

### Compare RMSE of y_test_scaled and y_pred_inverse
```
from tensorflow.keras.metrics import MeanSquaredError
import math
mse = MeanSquaredError()
print(math.sqrt(mse(y_test_scaled, y_pred_inverse)))
```

```
200358.04878267305
```

### Conclusion

The predicted MAE on the Test set is $186,325.62.

The MAE on the scaled Test set is 0.2344.

The real MAE is $184,885.94.

As the Test set has 292 patterns, the MAE of them is $186,325.62 * 292 = $52,406,900.

The RMSE of them is $200,358.05 * 292 = $58,504,536.

As I have used GridSearch to find the best model (optimizers, learning rate,...), I can not return its history plot. Only if I test the previous mode, I can plot the history then.

```
# Test the previous model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=X_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
my_optim=SGD(learning_rate = 0.01, momentum = 0.0)
model.compile(loss='mse', optimizer=my_optim, metrics=['mae',RootMeanSquaredError()])
history = model.fit(X_train_scaled, y_train_scaled, epochs=80)
print(history.history.keys())
plt.plot(history.history['loss']) 
```

```
Epoch 80/80
37/37 [==============================] - 0s 1ms/step - loss: 0.0801 - mae: 0.1975 - root_mean_squared_error: 0.2830
dict_keys(['loss', 'mae', 'root_mean_squared_error'])
```

<img src="https://user-images.githubusercontent.com/70437668/141609435-2b450e04-9e85-478b-adac-9e35af51aacd.jpg" width=50% height=50%>

As per the plot (x: epochs, y: loss), it has reached the max limit. Even if I keep training it, it will be quite the same for loss, mae and mse.


### V. Visualization in Tableau:

Housing affordability by Building Type in the US

![Housing affordability by Building Type in the US](https://user-images.githubusercontent.com/70437668/138982197-44e4f247-fe42-49f7-8ab1-3f7edab1277b.jpg)

Housing affordability by House Style in the US

![Housing affordability by House Style in the US](https://user-images.githubusercontent.com/70437668/138982217-e2a0ddbc-0bfa-4f6b-9bab-31dde5022e54.jpg)

Housing payment by Sale Type  in the US

![Housing payment by Sale Type  in the US](https://user-images.githubusercontent.com/70437668/138982244-4db38716-61bc-46a1-a6eb-4cc9c8688bf4.jpg)

Housing payment by Sale Condition in the US

![Housing payment by Sale Condition in the US](https://user-images.githubusercontent.com/70437668/138982273-4c7cd4ff-ca9e-46e9-abed-a6395cc058bf.jpg)

House price's movement by different remodelling years in the US

![House price's movement by different remodelling years in the US](https://user-images.githubusercontent.com/70437668/138982284-428a1d10-811d-4834-8b70-375c414012e2.jpg)

House price's movement by different years sold in the US

![House price's movement by different years sold in the US](https://user-images.githubusercontent.com/70437668/138982304-9c843f7a-1728-4b13-99c4-94e176db7dec.jpg)

House price's trends by different area types - part 1

![House price's trends by different area types - part 1](https://user-images.githubusercontent.com/70437668/138982323-5b4c23b7-1a33-40f5-badb-6cde91d243f1.jpg)

House price's trends by different area types - part 2

![House price's trends by different area types - part 2](https://user-images.githubusercontent.com/70437668/138982332-8af4a4b1-b065-4e95-af73-f72e4e6f5aab.jpg)

Type of dwelling involved in the US house sale

![Type of dwelling involved in the US house sale](https://user-images.githubusercontent.com/70437668/138982345-7755ad8b-e860-4991-9062-d88c98dd2c5c.jpg)

Affects on US house price sold through the Financial Crisis

![Affects on US house price sold through the Financial Crisis](https://user-images.githubusercontent.com/70437668/138982358-80463128-0a33-4cdf-a7ae-b34bf761fd05.jpg)

General zoning classification of the US house sale

![General zoning classification of the US house sale](https://user-images.githubusercontent.com/70437668/138982369-8c3a4941-6e6f-4771-982f-5c1098636b8b.jpg)

Rates the overall material and finish of the house in the US

![Rates the overall material and finish of the house in the US](https://user-images.githubusercontent.com/70437668/138982382-39b6113a-440e-4fe3-ab31-475f11116063.jpg)

Original construction date of US houses

![Original construction date of US houses](https://user-images.githubusercontent.com/70437668/138982396-b6b37d01-56fc-4339-b3d9-8b9c3ce7be21.jpg)

'SalePrice' vs correlated variables (circle map) 

!['SalePrice' vs correlated variables (circle map) ](https://user-images.githubusercontent.com/70437668/138982409-8c2070a3-c832-4d5f-b359-0bc2b07073f8.jpg)

Dashboard - Housing Affordability by property & sale types in the US

![Dashboard - Housing Affordability by property   sale types in the US](https://user-images.githubusercontent.com/70437668/138982423-6b818f8a-8cf0-4396-a161-49d9c53fba28.jpg)

Dashboard - Housing Affordability by area types in the US

![Dashboard - Housing Affordability by area types in the US](https://user-images.githubusercontent.com/70437668/138982435-842fba18-1d15-4c67-b8db-31536e3c94ec.jpg)

Dashboard - Insights of US housing life cycle

![Dashboard - Insights of US housing life cycle](https://user-images.githubusercontent.com/70437668/138982459-cec4142c-d3e3-4c35-b0ba-bdfefd7e9503.jpg)

Dashboard - US house zonings & rates

![Dashboard - Insights of US housing life cycle](https://user-images.githubusercontent.com/70437668/138982474-f0a9f2ed-bbe1-4304-a703-c0b58a3d0584.jpg)

Story - Conclusion of the US house pricing market through the Financial Crisis 2007-2008

![Story - Conclusion of the US house pricing market through the Financial Crisis 2007-2008](https://user-images.githubusercontent.com/70437668/138982477-79fb65ab-2452-44b3-ba2a-321261b027c8.jpg)

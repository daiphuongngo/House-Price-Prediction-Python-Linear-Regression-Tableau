# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:39:35 2021

@author: phuongdaingo
"""

#%% Import Libraries
import time
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
np.set_printoptions(precision=3)
PATH = "/Users/phuongdaingo/Documents/Python/PythonFrankie/Package1/Final Term"


#%% PREPROCESSING 
df = pd.read_csv(PATH + "/train.csv") 
df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1) 

temp = df.columns.tolist() # list out all columns names 


#%% SalesPrices
df['SalePrice'].dtype  
df['SalePrice'].describe() #describe shows 1459 rows while df has 1460 rows. So there is 1 null. 
df['SalePrice'] = df['SalePrice'] * 1000 

plt.figure(figsize=(7, 5)) 
sns.distplot(df["SalePrice"]) 
plt.xlabel('Sale Price')
plt.title('Distplot chart of Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/SalePrice_distplot.png", dpi=100)


#%% MSSubClass Identifies the type of dwelling involved in the sale
df['MSSubClass'].dtype  
df['MSSubClass'].corr(df['SalePrice']) 
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MSSubClass'].describe()

df["MSSubClass"] = df["MSSubClass"].replace({"20": "1-STORY 1946" + "\n" + "& NEWER ALL STYLES", 
                              "30": "1-STORY 1945 & OLDER", 
                              "40": "1-STORY W/FINISHED" + "\n" + "ATTIC ALL AGES",
                              "45": "1-1/2 STORY UNFINISHED" + "\n" + "ALL AGES",
                              "50": "1-1/2 STORY FINISHED" + "\n" + "ALL AGES",
                              "60": "2-STORY 1946 & NEWER",
                              "70": "2-STORY 1945 & OLDER",
                              "75": "2-1/2 STORY ALL AGES",
                              "80": "SPLIT OR MULTI-LEVEL",
                              "85": "SPLIT FOYER",
                              "90": "DUPLEX" + "\n" + "ALL STYLES & AGES",
                              "120": "1-STORY PUD (Planned" + "\n" + "Unit Development)" + "\n" + "1946 & NEWER",
                              "150": "1-1/2 STORY" + "\n" + "PUD ALL AGES",
                              "160": "2-STORY" + "\n" + "PUD 1946 & NEWER",
                              "180": "PUD MULTILEVEL" + "\n" + "INCL SPLIT LEV/FOYER",
                              "190": "2 FAMILY CONVERSION" + "\n" + "ALL STYLES AND AGES"})


plt.figure(figsize=(16, 18)) 
sns.boxplot(x='MSSubClass', y="SalePrice", data=df)
plt.ylim(0, 400000)
plt.xticks(rotation=90)
plt.xlabel('MS Sub Class: Identifies the type of dwelling involved in the sale.')
plt.title('Boxplot chart of MS Sub Class vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/MS Sub Class_boxplot.png", dpi=100)


#%% MSZoning
df['MSZoning'].dtype  
df['MSZoning'].describe()

df["MSZoning"] = df["MSZoning"].replace({"RL": "Residential" + "\n" + "Low Density", 
                              "RM": "Residential" + "\n" + "Medium Density", 
                              "C (all)": "Other Areas",
                              "FV": "Residential" + "\n" + "Floating Village",
                              "RH": "Residential" + "\n" + "High Density"})

plt.figure(figsize=(16, 16)) 
sns.boxplot(x='MSZoning', y="SalePrice", data=df)
plt.ylim(0, 500000)
plt.xticks(rotation=90)
plt.xlabel('MSZoning - Identifies the general zoning classification of the sale.')
plt.title('Boxplot chart of MS Zoning vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/MSZoning_boxplot.png", dpi=100)


#%% LotArea
df['LotArea'].dtype  
df['LotArea'].describe()
df['LotArea'].corr(df['SalePrice'])

sns.lmplot(data=df, x="LotArea", y="SalePrice",  hue = 'MSZoning') 
plt.ticklabel_format(style='plain') 
#sns.lmplot(data=df, x="LotArea", y="SalePrice") 
plt.title('lmplot chart of LotArea vs SalePrice from 2006 to 2010')
df['LotArea'].corr(df['SalePrice']) 
plt.xlabel('Lot Area')
plt.savefig(os.getcwd() + "/Package1/Picture/LotArea_lmplot.png", dpi=100)

sns.scatterplot(x="LotArea", y="SalePrice", data=df, hue="MSZoning",
                palette="ch:r=-.2,d=.3_r")
plt.ylim(0, 800000)
df['LotArea'].corr(df['SalePrice']) 
plt.title('lmplot chart of LotArea vs SalePrice from 2006 to 2010')
plt.xlabel('Lot Area')
plt.savefig(os.getcwd() + "/Package1/Picture/LotArea_scatter.png", dpi=100) 


#%% BldgType - Building type
df['BldgType'].dtype
df['BldgType'].value_counts()

df['BldgType'] = df['BldgType'].replace({"1Fam": "Single-family Detached",
                                         "2fmCon": "Two-family Conversion",
                                         "TwnhsE": "Townhouse End Unit",
                                         "Twnhs": "Townhouse Inside Unit"})

plt.figure(figsize=(16, 9)) 
sns.boxplot(data=df, x="BldgType", y="SalePrice", fliersize=0)
plt.ylim(0, 400000)
plt.xticks(rotation=0)
plt.xlabel('Building Type')
plt.title('Boxplot chart of Building Type vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/BuildingType_boxplot.png", dpi=100)

#%% HouseStyle
df['HouseStyle'].dtype
df['HouseStyle'].value_counts()
df['HouseStyle'] = df['HouseStyle'].replace({"1Story": "1 fl", "2Story": "2 fl",
                                             "1.5Fin": "1.5 fl: Lv2 finished",
                                             "SLvl": "Split Level", "SFoyer": "Split Foyer",
                                             "1.5Unf": "1.5 fl: Lv2 unfinished",
                                             "2.5Unf": "2.5 fl: Lv2 unfinished",
                                             "2.5Fin": "2.5 fl: Lv2 finished"})

plt.figure(figsize=(16, 8)) 
sns.boxplot(data=df, x="HouseStyle", y="SalePrice", fliersize=0)
plt.ylim(0, 400000)
plt.xticks(rotation=0)
plt.xlabel('House Style')
plt.title('Boxplot chart of House Style vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/HouseStyle_boxplot.png", dpi=100)


#%% YearBuilt 
df['YearBuilt'].dtype  
df['YearBuilt'].describe()
df['YearBuilt'].corr(df['SalePrice'])

plt.figure(figsize=(20, 10))
sns.boxplot(x='YearBuilt', y="SalePrice", data=df)
plt.ylim(0, 800000)
plt.xticks(rotation=90)
plt.xlabel('Year Built')
plt.title('Boxplot chart of YearBuilt vs SalePrice')
plt.savefig(os.getcwd() + "/Package1/Picture/YearBuilt_boxplot.png", dpi=100)

plt.figure(figsize=(20, 10))
sns.lmplot(data=df, x="YearBuilt", y="SalePrice") 
df['YearBuilt'].corr(df['YearBuilt'])
plt.xlabel('Year Built')
plt.title('lmplot chart of YearBuilt vs SalePrice')
plt.savefig(os.getcwd() + "/Package1/Picture/YearBuilt_lmplot.png", dpi=100)


#%% YrSold
df['YrSold'].dtype  
df['YrSold'].describe()
#df['SalesPrice'] = df['SalePrice'] / 1000
df['YrSold'].corr(df['SalePrice'])

plt.figure(figsize=(16, 8))
sns.boxplot(x='YrSold', y="SalePrice", data=df)
plt.ylim(0, 400000)
plt.xticks(rotation=0)
plt.xlabel('Year Sold')
plt.title('Boxplot chart of Year Sold vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/YearSold.png", dpi=100)


#%% SaleType
df['SaleType'].dtype  
df['SaleType'].describe()
#df = df.sort_values(by = ['SaleType'], axis=0, ascending=False)

df["SaleType"] = df["SaleType"].replace({"WD": "Warranty Deed" + "\n" + "Conventional", 
                              "CWD": "Warranty Deed" + "\n" + "Cash", 
                              "VWD": "Warranty Deed" + "\n" + "VA Loan",
                              "New": "Home just" + "\n" + "constructed & sold",
                              "COD": "Court Officer" + "\n" + "Deed/Estate",
                              "Con": "Contract 15% Down" + "\n" + "payment regular terms",
                              "ConLw": "Contract Low Down" + "\n" + "payment & low interest",
                              "ConLI": "Contract" + "\n" + "Low Interest",
                              "ConLD": "Contract" + "\n" + "Low Down",
                              "Oth": "Other"})

plt.figure(figsize=(20, 18))
sns.boxplot(data=df, x="SaleType", y="SalePrice", fliersize=0)
plt.ylim(0, 600000)
plt.xticks(rotation=90)
plt.xlabel('Sale Type')
plt.title('Boxplot chart of Sale Type vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/SaleType_boxplot.png", dpi=100)

plt.figure(figsize=(20, 18))
#sns.catplot(x="SaleType", y="SalePrice", data=df, kind="swarm")
sns.swarmplot(x="SaleType", y="SalePrice", data=df)
plt.ylim(0, 500000)
plt.xticks(rotation=90)
plt.xlabel('Sale Type')
plt.title('Swarmplot chart of Sale Type vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/SaleType_swarmplot.png", dpi=100)


#%% SaleCondition
df['SaleCondition'].dtype  
df['SaleCondition'].describe()

df["SaleCondition"] = df["SaleCondition"].replace({"Normal": "Normal Sale", 
                              "Abnorml": "Abnormal Sale", 
                              "AdjLand": "Adjoining" + "\n" + "Land Purchase",
                              "Alloca": "Allocation",
                              "Family": "CSale between" + "\n" + "family members",
                              "Partial": "Come wasn't completed" + "\n" + "when last assessed"})
    
plt.figure(figsize=(20, 18))
sns.boxplot(data=df, x="SaleCondition", y="SalePrice", fliersize=0)
plt.ylim(0, 600000)
plt.xticks(rotation=90)
plt.xlabel('Sale Condition')
plt.title('Boxplot chart of Sale Condition vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/SaleCondition_boxplot.png", dpi=100)

plt.figure(figsize=(20, 18))
#sns.catplot(x="SaleCondition", y="SalePrice", data=df, kind="swarm")
sns.swarmplot(x="SaleCondition", y="SalePrice", data=df)
plt.ylim(0, 500000)
plt.xticks(rotation=90)
plt.xlabel('Sale Condition')
plt.title('Catplot chart of Sale Condition vs Sale Price from 2006 to 2010')
plt.savefig(os.getcwd() + "/Package1/Picture/SaleCondition_catplot.png", dpi=100)


#%% OverallQual: Rates the overall material and finish of the house
df['OverallQual'].dtype  
df['OverallQual'].describe()

plt.figure(figsize=(16, 8))
sns.boxplot(x='OverallQual', y="SalePrice", data=df)
plt.ylim(0, 800000)
plt.xticks(rotation=0)
plt.xlabel('Overall Quality ')
plt.title('Boxplot chart of Overall Quality vs Sale Price from 2006 to 2010')
plt.legend() 
plt.savefig(os.getcwd() + "/Package1/Picture/Overall Quality.png", dpi=100)

df['OverallQual'].corr(df['SalePrice'])


#%% BUILD MODEL
def featuring():
    col = ['Id', 'SalePrice', 'MSSubClass', 'MSZoning', 'LotArea', 'BldgType', 
           'HouseStyle', 'YearBuilt', 'YrSold', 'SaleType', 'SaleCondition', 'OverallQual']
    df = df[col]
    
    dum_mssub = pd.get_dummies(df["MSSubClass"], prefix="mssub_")
    df = pd.merge(df, dum_mssub[:-1], left_index=True, right_index=True) #merge data and dummies of this column. Numbe of dummies = number of elements in this column. Merge by index (number of rows)
    
    dum_mszoning = pd.get_dummies(df["MSZoning"], prefix="mszoning_")
    df = pd.merge(df, dum_mszoning[:-1], left_index=True, right_index=True)
    
    dum_buildingtype = pd.get_dummies(df["BldgType"], prefix="buildingtype_")
    df = pd.merge(df, dum_buildingtype[:-1], left_index=True, right_index=True)
    
    dum_housestyle = pd.get_dummies(df["HouseStyle"], prefix="housestyle_")
    df = pd.merge(df, dum_housestyle[:-1], left_index=True, right_index=True)
    
    dum_saletype = pd.get_dummies(df["SaleType"], prefix="saletype_")
    df = pd.merge(df, dum_saletype[:-1], left_index=True, right_index=True)
    
    dum_salecondition = pd.get_dummies(df["SaleCondition"], prefix="salecondition_")
    df = pd.merge(df, dum_salecondition[:-1], left_index=True, right_index=True)
    
    df.drop(["MSSubClass", "MSZoning", "BldgType", "HouseStyle",
             "SaleType", "SaleCondition"], 
             axis=1, inplace=True)
    return df

def predict():
    _path = "/Users/phuongdaingo/Documents/Python/PythonFrankie/Package1/Final Term/train.csv"
    df = pd.read_csv(_path)
    df = featuring(df)
    
    #--------------
    # Method 1 
    train = df.sample(frac=0.7)
    test = df[~df["Id"].isin(train["Id"])] # filter by Id
    x_train = train.iloc[:, 2:]
    y_train = train["SalePrice"]
    x_test = test.iloc[:, 2:]
    y_test = test["SalePrice"]

    model = LinearRegression()
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    y_test = y_test.reset_index(drop=True) 
#    y_predict = y_predict.astype(float)
#    y_test = y_test.astype(float)
    
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = math.sqrt(mse)
    
    return rmse
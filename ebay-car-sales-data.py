#The purpose of this project is twofold:
#1 to clean and analyze used car listing data
#2 to rehearse the uses of the jupyter notebook in line with pandas work


import pandas as pd
import numpy as np

autos = pd.read_csv('autos.csv', encoding='Latin-1') #other encodings: UTF-8 and Windows-1252
autos.info() 
autos.head()
#jupyter can render the first few and last few values of any pandas object 


autos.columns #prints array of existing column names

autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'ab_test',
       'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'num_photos', 'postal_code',
       'last_seen']
#note the edit of column names for cleaning purposes and streamlined reading
#Neat find: Elegant Python function to convert CamelCase to snake_case?
# def convert(name):
#     s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
#     return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
#(command / to comment out a block of code)

autos.head()
#To explain the changes we made and why: cleans code, makes column easier to read
#easier to read data, easier to work with 

autos.describe(include='all') #used to look at descriptive based statistics -- that is, all the column descriptions of 
#data we just reviewed in the header/heading in last line of code 
#seeing the columns in the i python notebook (jupyter) should allow us as data analysts/scientists to take note to clean up 
#as needed: note columns displaying insignificance, columns in need of further investigation, and data stored as txt needs 
#to be cleaned 

#DQ points out that price and odometer columns are stored as text --we then are to remove the non-numeric characters 
#and convert any columns possible to numeric dtypes. 
#1
autos["num_photos"].value_counts()
#shown in output in notebook to be zero 
#2
autos = autos.drop(["num_photos", "seller", "offer_type"], axis=1)
#seller and offer_type don't provide any valuable information
#drop in pandas requires axis to be 1 if dealing with columns, default is zero for index(header)
#3 for price & auto columns, cleaning is needed to deal with extra charactrs 
autos["price"] = (autos["price"]
                        .str.replace("$","") #removes the dollar sign
                        .str.replace(",","") #removes the comma accompanying the indication of a thousand or more in price
                        .astype(int)    #translation to ensure analytical accuracy
                        )
autos["price"].head()
#4
autos["odometer"] = (autos["odometer"]
                             .str.replace("km","")  #removes km value
                             .str.replace(",","")   #removes comma typically indicating thousand or more km
                             .astype(int)   #cleansweep to ensure type to be an integer
                             )
#5
autos.rename({"odometer": "odometer_km"}, axis=1, inplace=True)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html 
#for inplace, if value is true, boolean of the copy can be ignored 
#6
autos["odometer_km"].head()
#7  
autos["odometer_km"].value_counts()
print(autos["price"].unique().shape)
print(autos["price"].describe())
autos["price"].value_counts().head(20)

autos["price"].value_counts().sort_index(ascending=False).head(20)
#(see ipynb)

autos["price"].value_counts().sort_index(ascending=True).head(20)
#(see ipynb)

autos = autos[autos["price"].between(1,351000)] # zero dollar listings don't provide us with useful information
autos["price"].describe()                       #cap around 350k to avoid outliers swaying the average
#(see ipynb)              #on average a used car on ebay in Germany goes for around five thousand nine hundred dollars

autos[['date_crawled','ad_created','last_seen']][0:5]

(autos["date_crawled"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )


(autos["date_crawled"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_values()
        )
#The dataset we're finding is from excavation of ads in the months of March and April 2016


(autos["last_seen"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )
#Further sorting reveals it was updated on a daily basis from March the fifth through April the seventh

print(autos["ad_created"].str[:10].unique().shape)
(autos["ad_created"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )
#Months of ad creation fell into three categories roughly: within the same month the ad was found, 
#within one to two months of findings and some, a good handful, from close to a whole year before --possibly no longer
#monitored but still active 

autos["registration_year"].describe()

(~autos["registration_year"].between(1900,2016)).sum() / autos.shape[0]
#determining the percent of dataset with unreasonable values --those above the year of posting (2016)
#and those not abiding by the standardization of cars beginning somewhere in the 1900s as a commericial vehicle 
#(we are not seeking out prehistoric wagons in auction)
#we find inaccurate listings to be 3.88% of the data so we can eliminate them from our set

# Many ways to select rows in a dataframe that fall within a value range for a column.
# One of them: `Series.between()`
autos = autos[autos["registration_year"].between(1900,2016)]
autos["registration_year"].value_counts(normalize=True).head(10)
#We won't be organizing the years in order but from an eye's glance, the data reveals a more 
#reliable collection of cars from 1998 to 2007 at the time of auction in 2016. 
#Thus German used cars are most likely to be sold at the age from anywhere from nine to eighteen years old on ebay.
#This also may indicate that people tend to keep their cars for a long time --on average, nine years-- before considering
#resale. Although, maybe ebay specializes in cars slightly older than consumers who would trade-in a slightly newer model
#every (or every other) year. 

autos["brand"].value_counts(normalize=True)
#The first five are all German brands: volkswagen, bmw, opel, mercedes_benz and audi. They make up nearly 60% of the dataset.
#The following Ford is American based. Renault and Peugeot are French. Fiat - Italian and so forth.
brand_counts = autos["brand"].value_counts(normalize=True)
common_brands = brand_counts[brand_counts > .05].index
print(common_brands)


brand_mean_prices = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_price = brand_only["price"].mean()
    brand_mean_prices[brand] = int(mean_price)

brand_mean_prices
# we aggregated across brands to understand mean price. In the top 6 brands, there's a price gap where
# Audi, BMW and Mercedes Benz are more expensive than average
# Ford and Opel are less expensive than average
# Volkswagen was about average.
# For these brands, aggregation can be further used to understand the average mileage for these car brands and if there's true correlation with mean price.
# We shouldn't display both aggregated series objects and visually compare them due to limitations:
#       - it's difficult to compare more than two aggregate series objects if we were to create more columns
#       - we can't compare more than a few rows from each series object
#       - we can only sort by the index (brand name) of both series objects so we can easily make visual comparisons
# Instead, we will combine the data from both series objects into a single dataframe (with a shared index) and display the dataframe directly. 
# To do this, we'll need to learn two pandas methods:

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html 

bmp_series = pd.Series(brand_mean_prices)
pd.DataFrame(bmp_series, columns=["mean_mileage"])

brand_mean_mileage = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_mileage = brand_only["odometer_km"].mean()
    brand_mean_mileage[brand] = int(mean_mileage)

mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending=False)
mean_prices = pd.Series(brand_mean_prices).sort_values(ascending=False)

brand_info = pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_info

brand_info["mean_price"] = mean_prices
brand_info

# The range of car mileages does not vary as much as the prices do by brand, all have a mileage within 10% 
# for the brands analyzed. If anything, more expensive vehicles having higher mileage and 
# less expensive having lower, suggesting there is no significant difference in mileage to impact pricing.
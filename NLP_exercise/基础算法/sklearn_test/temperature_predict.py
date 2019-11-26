# -*- coding: utf-8 -*-

"""

	预测未来气候

"""

import pandas as pd
import numpy as np
from matplotlib.pylab import plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import io


# load dataset
url = "http://blog.topspeedsnail.com/wp-content/uploads/2016/12/GlobalTemperatures.csv"
climate_data = requests.get(url).content

climate_df = pd.read_csv(io.StringIO(climate_data.decode("utf-8")))

# convert date to datetime
climate_df.date = pd.to_datetime(climate_df.date)

# convert datetime to year, month, day
climate_df['year'] = climate_df['date'].dt.year
climate_df['month'] = climate_df['date'].dt.month
climate_df = climate_df.drop('date', 1)

 
# only use LandAverageTemperature attribute
climate_df = climate_df[np.isfinite(climate_df['LandAverageTemperature'])]

climate_df = climate_df.drop(['LandAverageTemperatureUncertainty'], 1)
climate_df = climate_df.drop(['LandMaxTemperature'], 1)
climate_df = climate_df.drop(['LandMaxTemperatureUncertainty'], 1)
climate_df = climate_df.drop(['LandMinTemperature'], 1)
climate_df = climate_df.drop(['LandMinTemperatureUncertainty'], 1)
climate_df = climate_df.drop(['LandAndOceanAverageTemperature'], 1)
climate_df = climate_df.drop(['LandAndOceanAverageTemperatureUncertainty'], 1)
climate_df = climate_df.fillna(-9999)


X = np.array(climate_df.drop(['LandAverageTemperature'], 1))
Y = np.array(climate_df['LandAverageTemperature'])


clf = GradientBoostingRegressor(learning_rate = 0.03, max_features = 0.03, n_estimators = 500)
clf.fit(X, Y)

predict_x = []
start_year = 2016

for year in range(3):
	for month in range(12):
		predict_x.append([start_year + year, month + 1])


# predict 2016 ~ 2025 year's temperature
predict_y = clf.predict(predict_x) 


# draw 1980 ~ 2015 year's average temperature
year_x = []
for x in X:
	year_x.append(x[0])

data = {}


for x, y in zip(year_x, Y):
	if x not in data.keys():
		data[x] = y
	else:
		data[x] = (data[x] + y)

for key, value in data.items():
	if key > 1980:
		plt.scatter(key, value / 12)

plt.show()


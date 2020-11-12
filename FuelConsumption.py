
# Regression 

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

def simple_linear_regression():
	# reading in the file
	df = pd.read_csv("FuelConsumption.csv")

	# take a look at the dataset
	df.head()

	# plotting each feature in a histogram
	viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
	viz.hist()
	plt.show()

	# plotting consumption vs emissions
	plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
	plt.xlabel("FUELCONSUMPTION_COMB")
	plt.ylabel("Emission")
	plt.show()

	# modeling data
	from sklearn import linear_model
	regr = linear_model.LinearRegression()
	train_x = np.asanyarray(train[['ENGINESIZE']])
	train_y = np.asanyarray(train[['CO2EMISSIONS']])
	regr.fit (train_x, train_y)
	# The coefficients
	print ('Coefficients: ', regr.coef_)
	print ('Intercept: ',regr.intercept_)

	# plotting the fit line
	plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
	plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
	plt.xlabel("Engine size")
	plt.ylabel("Emission")

	# obtaining evaluation metrics
	from sklearn.metrics import r2_score

	test_x = np.asanyarray(test[['ENGINESIZE']])
	test_y = np.asanyarray(test[['CO2EMISSIONS']])
	test_y_hat = regr.predict(test_x)

	print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
	print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
	print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

def multiple_linear_regression():
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn import linear_model
	train_x = np.asanyarray(train[['ENGINESIZE']])
	train_y = np.asanyarray(train[['CO2EMISSIONS']])

	test_x = np.asanyarray(test[['ENGINESIZE']])
	test_y = np.asanyarray(test[['CO2EMISSIONS']])


	poly = PolynomialFeatures(degree=2)
	train_x_poly = poly.fit_transform(train_x)
	train_x_poly

	# solve using linear regression
	clf = linear_model.LinearRegression()
	train_y_ = clf.fit(train_x_poly, train_y)

	# The coefficients
	print ('Coefficients: ', clf.coef_)
	print ('Intercept: ',clf.intercept_)

	# plotting
	plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
	XX = np.arange(0.0, 10.0, 0.1)
	yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
	plt.plot(XX, yy, '-r' )
	plt.xlabel("Engine size")
	plt.ylabel("Emission")

	# obtaining evaluation metrics
	from sklearn.metrics import r2_score
	test_x_poly = poly.fit_transform(test_x)
	test_y_ = clf.predict(test_x_poly)

	print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
	print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
	print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

	######### COMPLETE WITH DEGREE 3 ############
	poly3 = PolynomialFeatures(degree=3)
	train_x_poly3 = poly3.fit_transform(train_x)
	clf3 = linear_model.LinearRegression()
	train_y3_ = clf3.fit(train_x_poly3, train_y)
	# The coefficients
	print ('Coefficients: ', clf3.coef_)
	print ('Intercept: ',clf3.intercept_)
	plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
	XX = np.arange(0.0, 10.0, 0.1)
	yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
	plt.plot(XX, yy, '-r' )
	plt.xlabel("Engine size")
	plt.ylabel("Emission")
	test_x_poly3 = poly3.fit_transform(test_x)
	test_y3_ = clf3.predict(test_x_poly3)
	print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
	print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
	print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )
	######### COMPLETE WITH DEGREE 3 ############

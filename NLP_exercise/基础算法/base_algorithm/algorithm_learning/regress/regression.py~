# -*- coding: utf-8 -*-

from numpy import *

def load_dataSet(filename):
	num_feature = len(open(filename).readline().split('\t')) - 1
	data_matrix = []
	label_matrix = []
	file = open(filename)

	for line in file.readlines():
		line_array = []
		current_line = line.strip().split('\t')

		for i in range(num_feature):
			line_array.append(float(current_line[i]))
		data_matrix.append(line_array)
		label_matrix.append(float(current_line[-1]))
	return data_matrix, label_matrix



def stand_regress(x_array, y_array):
	x_matrix = mat(x_array)
	y_matrix = mat(y_array).T
	xTx = x_matrix.T * x_matrix

	if linalg.det(xTx) == 0.0:
		print "the xTx matrix is singular, cannot do inverse"
		return

	weights = xTx.I * x_matrix.T * y_matrix
	return weights



def plot_regress(x_array, y_array, weights):
	import matplotlib.pyplot as plt

	x_matrix = mat(x_array)
	y_matrix = mat(y_array)
	y_hat = x_matrix * weights

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.scatter(x_matrix[:, 1].flatten().A[0], y_matrix.T[:, 0].flatten().A[0])

	x_copy = x_matrix.copy()
	x_copy.sort(0)
	y_hat = x_copy * weights

	ax.plot(x_copy[:, 1], y_hat)
	plt.show()



# 局部加权线性回归（Locally Weightsed Linear Regression）
# 给待预测点附近的每个点赋予一定的权重，计算出来的回归系数为：
#	w = (X.T * W * X)^(-1) * X.T * W * Y
# 其中，W为权重
# LWLR 使用"核"机制来对附近的点赋予更高的权重

def lwlr(test_point, x_array, y_array, k = 1.0):
	x_matrix = mat(x_array)
	y_matrix = mat(y_array).T

	row = shape(x_matrix)[0]
	
	#样本权重矩阵
	weights = mat(eye(row))

	for j in range(row):
		diff_matrix = test_point - x_matrix[j, :]

		# 使用高斯核来更新权值
		weights[j, j] = exp(diff_matrix * diff_matrix.T / (-2.0 * k**2))

	xTx = x_matrix.T * (weights) * x_matrix

	if linalg.det(xTx) == 0:
		print "the matrix is singular, cannot do inverse"
		return
	
	ws = xTx.I * x_matrix.T * weights * y_matrix

	return test_point * ws


def lwlr_test(test_array, x_array, y_array, k = 1.0):
	row = shape(test_array)[0]

	y_hat = zeros(row)

	for i in range(row):
		y_hat[i] = lwlr(test_array[i], x_array, y_array, k)
	return y_hat



def regress_error(y_array, y_hatArray):
	return ((y_array - y_hatArray)**2).sum()



# 岭回归（ridge regression）
# 当样本点小于特征值时，则 X.T * X 为 singular matrix
# 故岭回归是指在上述矩阵中加上一个 lambda * I， 其中 I 为单位阵，lambda 为常数

def ridge_regress(x_matrix, y_matrix, lam = 0.2):
	xTx = x_matrix.T * x_matrix
	denom = xTx + eye(shape(x_matrix)[1]) * lam

	if linalg.det(denom) == 0.0:
		print "the matrix is singulat, cannot da inverse"
		return

	ws = denom.I * x_matrix.T * y_matrix
	return ws


def ridge_test(x_array, y_array):
	x_matrix = mat(x_array)
	y_matrix = mat(y_array).T

	x_mean = mean(x_matrix, 0)
	x_var = var(x_matrix, 0)
	x_matrix = (x_matrix - x_mean) / x_var

	y_mean = mean(y_matrix, 0)
	y_matrix = y_matrix - y_mean

	num_test = 30
	weights_matrix = zeros((num_test, shape(x_matrix)[1]))

	for i in range(num_test):
		ws = ridge_regress(x_matrix, y_matrix, exp(i - 10))
		weights_matrix[i, :] = ws.T
	return weights_matrix




# 前向逐步线性回归（贪婪策略：每一步都尽可能地减少误差， 首先所有样本权重设为1， 然后每步所做的决策是对某个权重增加或减少一个很小的值）

def stage_wise(x_array, y_array, steps = 0.01, num_iter = 100):
	x_matrix = mat(x_array)
	y_matrix = mat(y_array).T

	y_mean = mean(y_matrix, 0)
	y_matrix = y_matrix - y_mean

	# x_matrix = regularize(x_matrix)

	x_mean = mean(x_matrix, 0)
	x_var = var(x_matrix, 0)
	x_matrix = (x_matrix - x_mean) / x_var

	row, column = shape(x_matrix)
	return_matrix = zeros((num_iter, column))

	ws = zeros((column, 1))
	ws_test = ws.copy()
	ws_max = ws.copy()

	for i in range(num_iter):
		print ws.T
		lowest_error = inf

		for j in range(column):
			for sign in [-1, 1]:
				ws_test = ws.copy()
				ws_test[j] += steps * sign
				y_test = x_matrix * ws_test
				current_error = regress_error(y_matrix.A, y_test.A)

				if current_error < lowest_error:
					lowest_error = current_error
					ws_max = ws_test

		ws = ws_max.copy()
		return_matrix[i, :] = ws.T
	return return_matrix

'''

from time import sleep
import json
import urllib2

def search_forSet(ret_x, ret_y, set_number, year_time, num_pce, original_price):
	sleep(10)
	myAPI_str = 'get from code.google.com'
	search_URL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPI_str, set_number)

	page = urllib2.urlopen(search_URL)
	ret_dict = json.loads(page.read())

	for i in range(len(ret_dict['items'])):
		try:
			current_item = ret_dict['item'][i]

			if current_item['product']['condition'] == 'new':
				new_flag = 1
			else:
				new_flag = 0

			list_inv = current_item['product']['inventories']

			for item in list_inv:
				sell_price = item['price']
				
				if sell_price > original_price * 0.5:
					print "%d\t%d\t%d\t%f\t%f" %(year_time, num_pce, new_flag, original_price, sell_price)

					ret_x.append([year_time, num_pce, new_flag, original_price])
					ret_y.append(sell_price)
		except:
			print "problem with item %d " %i

'''


def scrapePage(inFile,outFile,yr,numPce,origPrc):
    from BeautifulSoup import BeautifulSoup
    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
    soup = BeautifulSoup(fr.read())
    i=1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print "item #%d did not sell" % i
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
            print "%s\t%d\t%s" % (priceStr,newFlag,title)
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()



def setDataCollect():
    scrapePage('setHtml/lego8288.html','out.txt', 2006, 800, 49.99)
    scrapePage('setHtml/lego10030.html','out.txt', 2002, 3096, 269.99)
    scrapePage('setHtml/lego10179.html','out.txt', 2007, 5195, 499.99)
    scrapePage('setHtml/lego10181.html','out.txt', 2007, 3428, 199.99)
    scrapePage('setHtml/lego10189.html','out.txt', 2008, 5922, 299.99)
    scrapePage('setHtml/lego10196.html','out.txt', 2009, 3263, 249.99)



'''
def set_dataCollect(ret_x, ret_y):
	search_forSet(ret_x, ret_y, 8288, 2006, 800, 49.99)
	search_forSet(ret_x, ret_y, 10030, 2002, 3096, 269.99)
	search_forSet(ret_x, ret_y, 10179, 2007, 5195, 499.99)
			

	search_forSet(ret_x, ret_y, 10181, 2007, 3428, 199.99)
	search_forSet(ret_x, ret_y, 10189, 2008, 5922, 299.99)
	search_forSet(ret_x, ret_y, 10196, 2009, 3263, 249.99)

'''













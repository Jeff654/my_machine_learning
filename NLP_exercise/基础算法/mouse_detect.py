# -*- coding: utf-8 -*-

import pyautogui
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


degree = 5
model = make_pipeline(PolynomialFeatures(degree), Ridge())

def predict_mouse_pos(mouse_pos):
	if len(mouse_pos) == 30:
		X = []
		Y = []
		for x, y in mouse_pos:
			X.append(x)
			Y.append(y)

		X = np.array(X).reshape(-1, 1)
		Y = np.array(Y).reshape(-1, 1)

		model.fit(X, Y)
		
		if mouse_pos[-1][0] >= mouse_pos[-2][0]:
			y_ = model.predict(np.array(mouse_pos[-1][0] + 10).reshape(-1, 1))
			return mouse_pos[-1][0] + 10, y_

		elif mouse_pos[-1][0] < mouse_pos[-2][0]:
			y_ = model.predict(np.array(mouse_pos[-1][0] - 10).reshape(-1, 1))
			return mouse_pos[-1][0] - 10, y_

	else:
		return None, None


# save a few closest position
mouse_pos = []
pre_pos = pyautogui.position()

while True:
	# the mouse not move
	if pyautogui.position() == pre_pos:
		pass

	# the mouse moves
	else:
		x, y = pyautogui.position()
		pre_pos = (x, y)
		mouse_pos.append((x, y))

		print('-' * 40)
		print('the current position is: ', x, y)
		
		predict_x, predict_y = predict_mouse_pos(mouse_pos)

		print('the prediction position of mouse is: ', predict_x, predict_y)
		print('-' * 40)

		if len(mouse_pos) == 30:
			del mouse_pos[0]




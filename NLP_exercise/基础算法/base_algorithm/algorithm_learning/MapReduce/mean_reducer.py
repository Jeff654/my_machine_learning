# -*- coding: utf-8 -*-

import sys
from numpy import *

def read_input(file):
	for line in file:
		yield line.rstrip()

input = read_input(sys.stdin)
mapper_out = [line.split('\t') for line in input]
print "mapper_out: ", mapper_out

cum_value = 0.0
cum_sumSquare = 0.0
cum_number = 0

for instance in mapper_out:
	j = float(instance[0])
	cum_number += j
	cum_value += j * float(instance[1])
	cum_sumSquare += j * float(instance[2])
mean = cum_value / cum_number
var_sum = (cum_sumSquare - 2 * mean * cum_value + cum_number * mean * mean) / cum_number

print "%d \t %f \t %f" %(cum_value, mean, var_sum)
print >> sys.stderr, "report: still alive"














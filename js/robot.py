#!/usr/bin/python

#import PyV8
#import random
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import MySQLdb as mdb

def create_mdb(table):
	con = mdb.connect('localhost','root',' ','testdb')

	with con:
		cur = con.cursor()
		dropcommand = "DROP TABLE IF EXISTS " + str(table)
		cur.execute(dropcommand)
		createcommand = "CREATE TABLE " + str(table) + "(id INT NOT NULL AUTO_INCREMENT, "
		for i in range(16):
			createcommand = createcommand + "Tile_" + str(i) + " INT(5), "
		createcommand = createcommand + "Direction INT(1), PRIMARY KEY (id))"
		cur.execute(createcommand)
	print 'created table for training data"

def add2mdb(table,invec,output):
	con = mdb.connect('localhost','root',' ','testdb')

	with con:
		cur = con.cursor()
		command = "INSERT INTO " + str(table) + " VALUES(\'" + str(invec) + "\', \'"
		for i in range(16):
			command = command + str(invec[i]) + "\', \'"
		command = command + str(output) + "\')"
		cur.execute(command)

def readmdb(table):
	con = mdb.connect('localhost','root',' ','testdb')

	with con:
		cur = con.cursor()
		command = "SELECT * FROM " + str(table)
		cur.execute(command)
		rows = cur.fetchall()

	dict = {}
	for row in rows():
		values = []
		invec = row[1:17]
		out = row[18]

def checkmoves(invec):
	allowablemoves = np.zeros(4)
	allowablemoves[0] = check(invec,0)
	allowablemoves[1] = check(invec,1)
	allowablemoves[2] = check(invec,2)
	allowablemoves[3] = check(invec,3)
	return allowablemoves

def check(invec,direx):
	vecs = np.zeros((4,4))
	if direx == 0:
		vecs[0,:] = invec[[0,1,2,3]]
		vecs[1,:] = invec[[4,5,6,7]]
		vecs[2,:] = invec[[8,9,10,11]]
		vecs[3,:] = invec[[12,13,14,15]]
	elif direx == 1:
		vecs[0,:] = invec[[0,4,8,12]]
		vecs[1,:] = invec[[1,5,9,13]]
		vecs[2,:] = invec[[2,6,10,14]]
		vecs[3,:] = invec[[3,7,11,15]]
	elif direx == 2:
		vecs[0,:] = invec[[3,2,1,0]]
		vecs[1,:] = invec[[7,6,5,4]]
		vecs[2,:] = invec[[11,10,9,8]]
		vecs[3,:] = invec[[15,14,13,12]]
	elif direx == 3:
		vecs[0,:] = invec[[12,8,4,0]]
		vecs[1,:] = invec[[13,9,5,1]]
		vecs[2,:] = invec[[14,10,6,2]]
		vecs[3,:] = invec[[15,11,7,3]]

	for i in range(4):
		vec = vecs[i,:]
		if np.nonzero(vec)[0][0] > 0:
			return 1
		vec_nums = vec[np.nonzero(vec)]
		for j in range(len(vec_nums)-1):
			if vec_nums[j] == vec_nums[j+1]:			
				return 1
	return 0

def update_board(invec,direx):
	vecs = np.zeros((4,4))
	if direx == 0:
		vecs[0,:] = invec[[0,1,2,3]]
		vecs[1,:] = invec[[4,5,6,7]]
		vecs[2,:] = invec[[8,9,10,11]]
		vecs[3,:] = invec[[12,13,14,15]]
	elif direx == 1:
		vecs[0,:] = invec[[0,4,8,12]]
		vecs[1,:] = invec[[1,5,9,13]]
		vecs[2,:] = invec[[2,6,10,14]]
		vecs[3,:] = invec[[3,7,11,15]]
	elif direx == 2:
		vecs[0,:] = invec[[3,2,1,0]]
		vecs[1,:] = invec[[7,6,5,4]]
		vecs[2,:] = invec[[11,10,9,8]]
		vecs[3,:] = invec[[15,14,13,12]]
	elif direx == 3:
		vecs[0,:] = invec[[12,8,4,0]]
		vecs[1,:] = invec[[13,9,5,1]]
		vecs[2,:] = invec[[14,10,6,2]]
		vecs[3,:] = invec[[15,11,7,3]]

	for i in range(4):
		vec = vecs[i,:]
		newvec = np.zeros(4)
		vec_nums = vec[np.nonzero(vec)]
		j = 0
		newj = 0
		while j < len(vec_nums)-1:
			if vec_nums[j] == vec_nums[j+1]:
				newvec[newj] = vec_nums[j]*2
				j += 1
			j += 1
			newj += 1
		vecs[i,:] = newvecs

	if direx == 0:
		invec[[0,1,2,3]] = vecs[0,:]
		invec[[4,5,6,7]] = vecs[1,:]
		invec[[8,9,10,11]] = vecs[2,:]
		invec[[12,13,14,15]] = vecs[3,:]
	elif direx == 1:
		invec[[0,4,8,12]] = vecs[0,:]
		invec[[1,5,9,13]] = vecs[1,:]
		invec[[2,6,10,14]] = vecs[2,:]
		invec[[3,7,11,15]] = vecs[3,:]
	elif direx == 2:
		invec[[3,2,1,0]] = vecs[0,:]
		invec[[7,6,5,4]] = vecs[1,:]
		invec[[11,10,9,8]] = vecs[2,:]
		invec[[15,14,13,12]] = vecs[3,:]
	elif direx == 3:
		invec[[12,8,4,0]] = vecs[0,:]
		invec[[13,9,5,1]] = vecs[1,:]
		invec[[14,10,6,2]] = vecs[2,:]
		invec[[15,11,7,3]] = vecs[3,:]

	return invec

def keypress_learn(direx):
	#get user keypress

	#add to training database

	#get newtile from javascript
	newtilevalue = ?
	newtilepos = ?
	return newtilevalue, newtilepos

def keypress_play(direx):
	#tell javascript which direx to go...

	#get newtile from javascript
	newtilevalue = ?
	newtilepos = ?
	return newtilevalue, newtilepos
	
def convert_output(output):
	ntrain = output.shape[0]
	outarray = np.zeros((ntrain,4))
	for i in range(ntrain):
		outarray[i,output[i]-1] = 1
	return outarray

def propagate(invec,weights1,weights2):
	z = np.dot(np.transpose(weights1),invec)
	activations = 1.0/(1+exp(-z))

	activations = np.append(0,activations)
	z = np.dot(np.transpose(weights2),activations)
	hypothesis = 1.0/(1+exp(-z))

	return activations, hypothesis

def backprop(inputs,outvec,weights1,weights2,act,hyp):
	delta3 = (hyp - outvec).reshape(4,1)
	delta2 = np.dot(weights2[1:25][:],delta3)*act[1:]*(1-act[1:])

	partial2 = act*np.transpose(delta3)
	partial1 = inputs*np.transpose(delta2)
	return partial1,partial2

def cost(weights,inputs,outarray,lam,notbias1,notbias2,notzeros1):
	weights1 = weights[:408].reshape(17,24)
	weights2 = weights[-100:].reshape(25,4)
	weights1 = notzeros1*weights1

	cost1 = 0.0
	delta1 = np.zeros((17,24))
	delta2 = np.zeros((25,4))
	for i in range(ntrain):
		#forward prop
		activations, hypothesis = propagate(inputs[i,:],weights1,weights2)
		cost1 += outarray[i,:]*log(transpose(hypothesis)) \\
			+ (1-outarray[:,i])*log(1-transpose(hypothesis))

		#back prop
		partial1,partial2 = backprop(inputs[i,:],outarray[i,:],weights1,weights2,activations,hypothesis)
		delta1 += partial1
		delta2 += partial2

	D1 = delta1/ntrain
	D1[1:,:] += lam*weights1[1:][:]
	D2 = delta2/ntrain
	D2[1:,:] += lam*weights2[1:][:]
	deriv = np.append(D1,D2)

	cost2 = np.sum((notbias1*weights1)**2) + np.sum((notbias2*weights2)**2)
	cost = -cost1/ntrain + lam*cost2/(2*ntrain)

	#weights1,weights2 = train(inputs[i,:],weights1,weights2,outarray[i,:])

	#use grad checking to compare partials using back prop vs numerical estimate of grad of cost function
	#then disable grad checking code
	return cost, deriv

def trainmode():
	#given training set inputs (array) and output (vector)...
	inputs, output = readmdb('Training')
	ntrain = inputs.shape[0]
	inputs = np.hstack((np.zeros((ntrain,1)),inputs)) # (ntrain x 17), training data in rows
	outarray = convert_output(output) # (ntrain x 4)
	lam = 1000
	twoeps = 0.2

	notbias1 = np.ones((17,24))
	notbias1[0,:] = np.zeros(24)

	notbias2 = np.ones((25,4))
	notbias2[0,:] = np.zeros(4)

	notzeros1 = np.ones((17,24))
	notzeros1[[5,6,7,8,9,10,11,12,13,14,15,16],16] = np.zeros(12)
	notzeros1[[1,2,3,4,9,10,11,12,13,14,15,16],17] = np.zeros(12)
	notzeros1[[1,2,3,4,5,6,7,8,13,14,15,16],18] = np.zeros(12)
	notzeros1[[1,2,3,4,5,6,7,8,9,10,11,12],19] = np.zeros(12)
	notzeros1[[2,3,4,6,7,8,10,11,12,14,15,16],20] = np.zeros(12)
	notzeros1[[1,3,4,5,7,8,9,11,12,13,15,16],21] = np.zeros(12)
	notzeros1[[1,2,4,5,6,8,9,10,12,13,14,16],22] = np.zeros(12)
	notzeros1[[1,2,3,5,6,7,9,10,11,13,14,15],23] = np.zeros(12)

	weights = twoeps*(np.random.rand(508)-0.5)

	weights, cost, Dvec = fmin_l_bfgs_b(cost,weights,None,args=(inputs,outarray,lam,notbias1,notbias2,notzeros1))

	weights1 = weights[:408].reshape(17,24)
	weights2 = weights[-100:].reshape(25,4)
	weights1 = notzeros1*weights1

	weights = np.append(weights1,weights2)

	fileweights = open('weights.txt','w')
	fileweights.write(weights)

def playmode(inputs):
	fileweights = open('weights.txt','r')
	line = fileweights.readline()

	weights1 = np.zeros((17,24))
	weights2 = np.zeros((25,4))

	count = 0
	for i in range(17):
		for j in range(24):
			weights1[i,j] = float(line.split()[count])
			count += 1
	for i in range(25):
		for j in range(4):
			weights2[i,j] = float(line.split()[count])
			count += 1

	inputs = ...?
	while playing...:
		allowablemoves = checkmoves(inputs)
		act,hyp = propagate(inputs,weights1,weights2)

		output = hyp*allowablemoves

		if output[0] == np.max(output):
			newtilevalue, newtilepos = keypress_play(0)
			inputs = update_board(inputs,0)
		elif output[1] == np.max(output):
			newtilevalue, newtilepos = keypress_play(1)
			inputs = update_board(inputs,1)
		elif output[2] == np.max(output):
			newtilevalue, newtilepos = keypress_play(2)
			inputs = update_board(inputs,2)
		elif output[3] == np.max(output):
			newtilevalue, newtilepos = keypress_play(3)
			inputs = update_board(inputs,3)
		inputs[newtilepos] = newtilevalue

def main():
	blahblah
	blahblah

if __name__ == '__main__':
	main()

#ctxt = PyV8.JSContext()
#ctxt.enter()
#ctxt.eval(open("keyboard_input_manager.js").read())
#ctxt.eval("var template = 'Javascript in Python is {{ opinion }}';")
#
#
#opinion = random.choice(["cool","great","nice","insane"])
#rendered = ctxt.eval("Mustache.to_html(template, { opinion: '%s' })" % (opinion, ))
#print rendered

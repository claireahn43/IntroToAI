from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

#Q5
pen = []
car = []
i = 0
while i < 5:
	nnet, accuracy = testPenData()
	pen.append(accuracy)
	nnett, accu = testCarData()
	car.append(accu)
	i += 1

print ('Pen: ', pen)
print ('Max: ', max(pen))
print ('Average: ', average(pen))
print ('Standard deviation: ', stDeviation(pen))

print ('Car: ', car)
print ('Max: ', max(car))
print ('Average: ', average(car))
print ('Standard deviation: ', stDeviation(car))

#Q6
pendict = {}
cardict = {}
perceptron = 0
while perceptron <= 40:
	pendict[perceptron] = []
	cardict[perceptron] = []
	i = 0
	while i < 5:
		nnet, accuracy = testPenData([perceptron])
		pendict[perceptron].append(accuracy)
		nnett, accu = testCarData([perceptron])
		cardict[perceptron].append(accu)
		i += 1
	perceptron += 5

for perceptron in pendict:
	current = pendict[perceptron]
	print ('Perceptron: ', perceptron)
	print ('Max: ', max(current))
	print ('Average: ', average(current))
	print ('Standard deviation: ', stDeviation(current))

for perceptron in cardict:
	current = cardict[perceptron]
	print ('Perceptron: ', perceptron)
	print ('Max: ', max(current))
	print ('Average: ', average(current))
	print ('Standard deviation: ', stDeviation(current))
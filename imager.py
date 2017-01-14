from PIL import Image

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer

img = Image.open("twitter.png")
pic = img.load()
x_size, y_size = img.size

INPUTS = 2
HIDDEN_NODES = 15
OUTPUTS = 3

net = buildNetwork(INPUTS, HIDDEN_NODES, HIDDEN_NODES, OUTPUTS, bias=True, outclass=SigmoidLayer)
ds = SupervisedDataSet(INPUTS, OUTPUTS)

for x in range(0, x_size): #Iterate through the columns
    for y in range(0, y_size): # Iterate through the rows
        val = pic[x,y] #get the RGB for that pixel 
        val = (val[0]/256, val[1]/256, val[2]/256) #normalize the values to be between 0 and 1. 

        ds.addSample((x,y), val) #x,y is the input and the RGB value is the output.


trainer = BackpropTrainer(net, ds)
error = 0
for i in range(0,5000): #train 5000 iterations
    error = trainer.train()
    print(error)
    if i % 5 == 0: #Every five iterations, let's update our predicted image.
        for x in range(0,x_size):
            for y in range(0,y_size):
                val = net.activate((x,y))
                val = (int(val[0]*256), int(val[1]*256), int(val[2]*256))
                pic[x,y] = val
        img.save("twitter-predicted.png","PNG")
        print(str(i)+". Saved new image")

#sol = net.activate(test)
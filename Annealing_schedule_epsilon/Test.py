from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import ModelTest3 as MT

reload(MT)

# Build a model with a 3-dimensional hidden layer
model = MT.build_model(15, print_loss=True)
accErr = MT.accuracyOfModel(model)
print "Error = %s and Accuracy Of model is %s:"  % (accErr[0], accErr[1])

#
# Plot the decision boundary
MT.plot_decision_boundary(lambda x: MT.predict(model, x))
plt.title("Decision Boundary for hidden layer size 15")


#Comment out the following code to run neural network for different 
# dimensions of hidden layer
'''
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = MT.build_model(nn_hdim,print_loss=True)
    MT.plot_decision_boundary(lambda x: MT.predict(model, x))
    accErr = MT.accuracyOfModel(model)
    print "Error = %s and Accuracy Of model is %s:"  % (accErr[0], accErr[1])
'''
plt.show()


# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import keras as k
import tensorflow as tf
import time

# Own files
from CANN import CANN
from CANN import subANNs as subANNs
from CANN import Outputs
from CANN import Inputs
from CANN import Helpers as help


#####
##
#####

# Switch to read pre-defined model from HDD (attention: when using a noisy data set the data import will create new/different noise)
readModelFromDisk = True

# Switch to change problem
problem = 1
"""
	0: Treloar
	1: Generalized Mooney Rivlin
	2: Generalized Mooney Rivlin with noise
"""

#####
## General
#####

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Obtain input data and problem specific settings
F, P11, extra, ds, problemName, numTens, numDir, batchSize, epochs, incomp = Inputs.defineProblem(problem)

# Create output folder 
outputFolder = Outputs.prepareOutputFolder('output/', problemName)

# Define ANN architecture

#####
## Create model and statistics
#####
# Create model
if type(extra) is np.ndarray:
	numExtra = extra.shape[1]
else:
	numExtra = 0

myCANN = CANN.CANN_wrapper(subANNs.Psi_layers_wrapper(), subANNs.dir_layers_wrapper(), subANNs.w_layers_wrapper(), numTens=numTens, numDir=numDir, numExtra=numExtra, incomp=incomp)
model_fit, model_full = myCANN()

# Output debugging information
Outputs.showModelSummary(model_full, numDir=numDir, outputFolder=outputFolder)
Outputs.plotModelGraph(model_full, outputFolder, numDir=numDir)

if readModelFromDisk==False:
	#####
	## Compile and fit model
	#####

	# Compile model
	model_fit.compile(
		optimizer = k.optimizers.Adam(learning_rate = 0.001),
		loss = 'mean_squared_error', 
		metrics = ['MeanSquaredError']
		)

	# Split data into training and validation
	train_in, train_out, val_in, val_out = help.trainValidationSplit(F, P11, extra, numExtra, trainSize=0.8, outputFolder=outputFolder)

	# Fit model
	startTime = time.time()
	his = model_fit.fit(
		train_in,
		train_out,
		verbose = 1,
		batch_size = batchSize,
		epochs = epochs,
		validation_data = (val_in, val_out),
		callbacks = [tf.keras.callbacks.ModelCheckpoint(
			outputFolder+'modelWeights.h5',
			monitor = 'val_loss',
			verbose = 1,
			save_best_only = True,
			mode = 'min'
			)]
		)
	endTime = time.time()
	print('\n\n\n       FITTING TIME: ' + help.timeString(endTime-startTime) + '\n\n')

	# Plot/save loss
	Outputs.plotLoss(his, outputFolder)
	Outputs.saveLoss(his, outputFolder)
	
# Reload best weights
model_fit  = help.loadWeights(model_fit , outputFolder)
model_full = help.loadWeights(model_full, outputFolder)

#####
## Visualization of results
#####

ds.predict(model_fit)
Outputs.saveMainCurves(ds, outputFolder)
Outputs.plotMainCurves(ds, outputFolder)

if problem == 1 or problem == 2: # Cannot be done for Treloar, since it's not based on an analytical strain energy
	Outputs.saveAndPlotErrors(ds, model_full, outputFolder, 55, 300, stepSizeI1=0.2, stepSizeI2=1.)
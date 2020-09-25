# Standard imports
import numpy as np
from sklearn.model_selection import train_test_split

# Own files
from CANN import Outputs



def timeString(t):
	"""
	Returns a formated string of a time period (format: __h __m __s; unnecessary leading parts [hours or minutes] are omitted for shorter time periods).

	Parameters
	----------
	t : float
		Duration of the time period in seconds.
		
	Returns
	-------
	string : string
		Formatted time (format: __h __m __s; unnecessary leading parts [hours or minutes] are omitted for short times).
		
	"""

	h = int(np.floor(t/60./60.                    ))
	m = int(np.floor(t/60.     - h*60.            ))
	s = int(np.floor(t         - h*60.*60. - m*60.))

	string = ''
	if h > 0:
		string += str(h) + 'h '
	if m > 0 or h > 0:
		string += str(m) + 'm '
	string += str(s) + 's'

	return string



def trainValidationSplit(F, P11, extra, extraSize, trainSize=0.8, outputFolder=False, randomState=None):
	"""
	Performs a random split of the input data into a training and a validation part. F, P11, and extra (if used) have to have the same length.

	Parameters
	----------
	F : array of floats
		Array of deformation gradients (each: dimension 3x3).
	P11 : array of floats
		Array of first components of the 1st Piola Kirchhoff stress tensor.
	extra : array of floats or []
		Array of extra features (each: dimension extraSize).
	extraSize : int
		Size of the additional feature vector.
	trainSize : float, optional
		Determines the relative size of the training set; within (0,1), defaults to 0.8.
	outputFolder : string or false, optional
		String determining the output folder. When set, a MATLAB file is written that states which data point is placed in which set.
	randomState : int, optional
		Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. See documentation of sklearn.model_selection.train_test_split for details.
		
	Returns
	-------
	train_in : array of floats
		Training inputs.
	train_out : array of floats
		Training outputs.
	val_in : array of floats
		Validation inputs.
	val_out : array of floats
		Validation outputs.
		
	"""
	
	# Create an index array (which will be split as well) in order to track which data point was put in which set (and in which order they've been put)
	indices = range(P11.shape[0])

	if extraSize == 0:
		train_F, val_F, train_P11, val_P11, train_indices, val_indices = train_test_split(F, P11, indices, train_size = trainSize, random_state=randomState)
		
		train_in = train_F
		val_in = val_F	
	else:
		train_F, val_F, train_P11, val_P11, train_extra, val_extra, train_indices, val_indices = train_test_split(F, P11, extra, indices, train_size = trainSize, random_state=randomState)
		
		train_in = [train_F, train_extra]
		val_in = [val_F, val_extra]

	train_out = train_P11
	val_out = val_P11
	
	# Exports the details of the split (which data point ended up in which set) to a MATLAB file for later use
	if isinstance(outputFolder, str):
		Outputs.saveTrainingValidationSplit(train_indices, val_indices, outputFolder)

	return train_in, train_out, val_in, val_out
	
	
	
def loadWeights(model, outputFolder):
	"""
	Loads weights into model.
	
	Note: Mainly serves the purpose to have a single line of code where the filename is specified for all cases.

	Parameters
	----------
	model : keras model
		Model to load weights into.
	outputFolder : string
		Path where data was written to.
		
	Returns
	-------
	model : keras model
		Model with loaded weights.
		
	"""
	
	model.load_weights(outputFolder+'modelWeights.h5')
	return model
# Standard imports
import keras as k
import tensorflow as tf



def dir_layers_wrapper():
	"""
	Returns a function that defines the sub-ANN architecture for the directional sub-ANN.

	Note: the unconventional way of using wrapper functions instead of functions directly serves the purpose to easily substitute sub-ANN architectures with another architecture or an analytical function. The latter is especially interesting for the Psi sub-ANN and hence consequently implemented for all sub-ANNs.
		
	Returns
	-------
	dir_layers : function
		Function that defines the directional sub-ANN.
		
	"""
	
	def dir_layers(extra):
		dir = k.layers.Dense(8, activation=tf.keras.activations.tanh)(extra)
		dir = k.layers.Dense(8, activation=tf.keras.activations.tanh)(dir)
		
		# Note: after the user-defined layers, the CANN architecture automatically adds the following to achieve the desired shape:
		# - A dense layer (numDir*3, activation=tf.keras.activations.tanh, use_bias=False, name='dir')
		# - A normalization to ensure unit vectors
		# This can be altered in the CANN architecture itself (function: dir_subANN)
		
		return dir
	return dir_layers



def w_layers_wrapper():	
	"""
	Returns a function that defines the sub-ANN architecture for the weighting factor sub-ANN.

	Note: the unconventional way of using wrapper functions instead of functions directly serves the purpose to easily substitute sub-ANN architectures with another architecture or an analytical function. The latter is especially interesting for the Psi sub-ANN and hence consequently implemented for all sub-ANNs.

	Returns
	-------
	w_layers : function
		Function that defines the weighting factor sub-ANN.
		
	"""

	def w_layers(extra):
		w = k.layers.Dense(8, activation=tf.keras.activations.sigmoid)(extra)
		w = k.layers.Dense(8, activation=tf.keras.activations.sigmoid)(w)
		
		# Note: after the user-defined layers, the CANN architecture automatically adds the following to achieve the desired shape:
		# - A dense layer (numTens*(numDir+1), activation=tf.keras.activations.sigmoid, use_bias=False, name='w')
		# - A normalization to ensure a sum of 1 for each group of weighting factors
		# This can be altered in the CANN architecture itself (function: w_subANN)

		return w
	return w_layers



def Psi_layers_wrapper():
	"""
	Returns a function that defines the sub-ANN architecture for the strain energy density function sub-ANN.

	Note: the unconventional way of using wrapper functions instead of functions directly serves the purpose to easily substitute sub-ANN architectures with another architecture or an analytical function. The latter is especially interesting for the Psi sub-ANN and hence consequently implemented for all sub-ANNs.

	Returns
	-------
	Psi_layers : function
		Function that defines the strain energy density function sub-ANN.
		
	"""

	def Psi_layers(I, J, III_C):
		# As long as you're trying to model incompressible materials, there is no point to include layers for III_C
		
		I = k.layers.Dense(8, activation=tf.keras.activations.softplus)(I)
		I = k.layers.Dense(8, activation=tf.keras.activations.softplus)(I)
		
		J = k.layers.Dense(8, activation=tf.keras.activations.softplus)(J)
		J = k.layers.Dense(8, activation=tf.keras.activations.softplus)(J)
		
		Psi = k.layers.concatenate([I, J])
		
		# Note: after the user-defined layers, the CANN architecture automatically adds the following to achieve the desired shape:
		# - A dense layer (1, activation="linear", use_bias=False, name='Psi')
		# This can be altered in the CANN architecture itself (function: Psi_subANN)

		return Psi
	return Psi_layers
# Standard imports
import keras as k
import tensorflow as tf

# Own files
from CANN import ContinuumMechanics as CM



def CANN_wrapper(Psi_layers, dir_layers, w_layers, numTens=1, numDir=0, numExtra=0, incomp=True, computePsiOnly=False):
	"""
	Returns a function that includes the entire CANN architecture including all required subANNs. The CANN is customized to the desired numbers of generalized structural tensors & preferred directions, the size of the extra feature vector, and the incompressibility assumption.

	Parameters
	----------
	Psi_layers : function
		Function that defines the layers for the strain energy density subANN or an analytical function as substitute.
	dir_layers : function
		Function that defines the layers for the directions subANN or an analytical function as substitute (*).
	w_layers : function
		Function that defines the layers for the weighting factors subANN or an analytical function as substitute (*).
	numTens : int, optional
		Number of generalized structural tensors to use (at least 1).
	numDir : int, optional
		Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).
	numExtra : int, optional
		Number of additional features per data point to incorporate (in an anisotropic case this needs to be at least 1; in case of isotropy this number can be 0 or higher).
	incomp : boolean, optional
		Switch to determine whether the material should be considered as incompressible or not.
	computePsiOnly : boolean, optional
		Switch to enable a pure computation of Psi without an ANN (only to be used when Psi is given as an analytical substitute function instead of layers).

	(*): Analytical substitutes for w and dir are not yet fully implemented (we did not yet run into a practical use case) but for consistency with Psi (where there is indeed a use case) all subANNs are fed in the same form.

	Returns
	-------
	CANN : function
		A function that returns the custom CANN.
		
	"""

	def unitVector(dir):
		"""
		Transforms the flat tensor that results from the directional subANN to a properly shaped tensor which is normalized (each vector is of length 1).

		Parameters
		----------
		dir : tensor [shape: (?, numDir * 3)]
			Flat tensor which results from the directional subANN layers.

		Returns
		-------
		dir : tensor [shape: (?, numDir, 3)]
			Reshaped tensor that represents the directional vectors in normalized form (each direction vector is a unit vector).
			
		"""

		dir = tf.keras.layers.Reshape([numDir, 3])(dir)
		
		length = tf.norm(dir, ord='euclidean', axis=2)
		dir = tf.divide(dir, tf.tile(tf.keras.backend.expand_dims(length,2), tf.constant([1,1,3])))
		
		return dir



	def summation(w):
		"""
		Transforms the flat tensor that results from the weighting subANN to a properly shaped tensor which is normalized (the weight in each grou sum to 1).
		
		Parameters
		----------
		w : tensor [shape: (?, numTens * (numDir+1))]
			Flat tensor which results from the weighting factor subANN layers.
	  
		Returns
		-------
		w : tensor [shape: (?, numTens, numDir+1)]
			Reshaped tensor that represents the weighting factors in normalized form (the weight in each grou sum to 1).
			
		"""

		w = tf.keras.layers.Reshape([numTens, numDir+1])(w)
		sum = tf.norm(w, ord=1, axis=2)
		w = tf.divide(w, tf.tile(tf.keras.backend.expand_dims(sum,2), tf.constant([1,1,numDir+1])))
		
		return w



	def dir_subANN(extra):
		"""
		Returns the layers for the subANN for directional vectors.
		
		Parameters
		----------
		extra : tensor
			Input layer of extra features.
	  
		Returns
		-------
		dir : tensor
			Layers for the subANN for directional vectors.
			
		"""

		dir = dir_layers(extra)
		dir = k.layers.Dense(numDir*3, activation=tf.keras.activations.tanh, use_bias=False, name='dir')(dir)
		
		dir = k.layers.Lambda(lambda dir: unitVector(dir), name='unitVector')(dir)

		return dir



	def w_subANN(extra):
		"""
		Returns the layers for the subANN for the weighting factors.
		
		Parameters
		----------
		extra : tensor
			Input layer of extra features.
	  
		Returns
		-------
		w : tensor
			Layers for the subANN for the weighting factors.
			
		"""

		w = w_layers(extra)
		w = k.layers.Dense(numTens*(numDir+1), activation=tf.keras.activations.sigmoid, use_bias=False, name='w')(w)
		
		w = k.layers.Lambda(lambda w: summation(w), name='unitSum')(w)

		return w



	def Psi_subANN(I, J, III_C, extra=False):
		"""
		Returns the layers for the subANN for the strain energy density function.
		
		Parameters
		----------
		I : tensor
			Input layer of the generalized invariants I.
		J : tensor
			Input layer of the generalized invariants J.
		III_C : tensor
			Input layer of the invariant III_C.
		extra : tensor or False, optional
			Input layer of extra features.
	  
		Returns
		-------
		Psi : tensor
			Layers for the subANN for the strain energy density function.
			
		"""

		if computePsiOnly == True:
			return Psi_layers(I, J, III_C)
			
		if numExtra > 0:
			I = k.layers.concatenate([I, extra])
			J = k.layers.concatenate([J, extra])
			III_C = k.layers.concatenate([III_C, extra])
		
		Psi = Psi_layers(I, J, III_C)
		Psi = k.layers.Dense(1, activation="linear", use_bias=False, name='Psi')(Psi)
		
		return Psi



	def CANN():
		"""
		Creates the entire CANN and returns the resulting keras models as output.
	  
		Returns
		-------
		model_fit : keras model
			A keras model with limited outputs (only _11 component of first Piola Kirchhoff stress tensor) to be used during the fitting process.
		model_full : keras model
			A keras model with all relevant outputs.
		
		"""
		
		I_in = tf.keras.Input(shape=(numTens,), name='I')
		J_in = tf.keras.Input(shape=(numTens,), name='J')
		III_C_in = tf.keras.Input(shape=(1,), name='III_C')
		if numExtra == 0:
			# Create model from strain-energy sub ANN
			Psi_ann = Psi_subANN(I_in, J_in, III_C_in)
			Psi_model = k.models.Model(inputs=[I_in, J_in, III_C_in], outputs=[Psi_ann], name='Psi')
			
			dir_model = []
			w_model = []

			if numDir != 0:
				print('\n\n\n\nNo extra variables are provided; numDir cannot be non-zero')
				exit()
		else:
			extra_in = tf.keras.Input(shape=(numExtra,), name='extra')

			# Create model from strain-energy sub ANN
			Psi_ann = Psi_subANN(I_in, J_in, III_C_in, extra_in)
			Psi_model = k.models.Model(inputs=[I_in, J_in, III_C_in, extra_in], outputs=[Psi_ann], name='Psi')

			if numDir > 0:
				# Create model from direction sub ANN
				dir_ann = dir_subANN(extra_in)
				dir_model = k.models.Model(inputs=extra_in, outputs=dir_ann, name='dir')

				# Create model from weights sub ANN
				w_ann = w_subANN(extra_in)
				w_model = k.models.Model(inputs=extra_in, outputs=w_ann, name='w')
			else:
				dir_model = []
				w_model = []

		# Continuum mechanics formulae before Psi
		F, extra, C, invariants_I, invariants_J, invariant_III_C, F_isoRef, C_isoRef, invariants_I_isoRef, invariants_J_isoRef, invariants_III_C_isoRef = CM.pre_Psi(numExtra, numTens, numDir, w_model, dir_model)
		
		# Strain-energy
		if numExtra == 0:
			Psi = Psi_model([invariants_I, invariants_J, invariant_III_C])
			Psi_isoRef = Psi_model([invariants_I_isoRef, invariants_J_isoRef, invariants_III_C_isoRef])
		else:	
			Psi = Psi_model([invariants_I, invariants_J, invariant_III_C, extra])
			Psi_isoRef = Psi_model([invariants_I_isoRef, invariants_J_isoRef, invariants_III_C_isoRef, extra])
			
		Psi = tf.math.subtract(Psi, Psi_isoRef) # to ensure Psi=0 in reference configuration

		# Continuum mechanics formulae after Psi	
		if incomp==True:
			P11, P, S, sigma = CM.post_Psi_incomp(Psi, Psi_isoRef, F, F_isoRef)
		else:
			P11, P, S, sigma = CM.post_Psi(Psi, F)
		
		# Create and return models
		if numExtra == 0:
			inputs = F
		else:
			inputs = [F, extra]
		model_fit  = k.models.Model(inputs, P11) # The one for fitting (only P11 as output in order to not require a complicated loss function)
		model_full = k.models.Model(inputs, [Psi, P, S, sigma]) # The full one with all values of interest
		
		return model_fit, model_full
		
	return CANN
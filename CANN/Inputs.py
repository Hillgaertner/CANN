# Standard imports
import numpy as np
import tensorflow as tf
import keras as k

# Own files
from CANN import ContinuumMechanics as CM
from CANN import CANN
from CANN import subANNs as subANNs
from CANN import Helpers as help

"""
	NOTE:
	All functions and classes in this file are dedicated to read/create input data and define
	a problem to be solved by the CANN:
	
	function : defineProblem
		Can be used to store pre-defined problem definitions. Is called from the main file.
	class : ExternalData
		Designated to read any user-made text file that provides the data points which the CANN
		should use. Needs to be tailored to the input file of interest.
	class: DataSource
		subclass : StrainEnergy
		subclass : Tabular
			subclass : Treloar
		Designated to create the three main curves (uniaxial tension, pure shear and biaxial tension)
		either artificially from an analytical strain energy function or by interpolation from curve data.
		The example from Treloar's data is provided, others can be added.
	class: curve
		Helping class for managing curves

"""


######################
######################
### FUNCTION: DEFINE PROBLEM
######################
######################

def defineProblem(problem):
	"""
	Returns all relevant information (input data and settings) for a specific pre-defined problem.

	Parameters
	----------
	problem : int
		problem ID.

	Returns
	-------
	F : array of floats
		Array of deformation gradients (each: dimension 3x3).
	P11 : array of floats
		Array of first components of the 1st Piola Kirchhoff stress tensor.
	extra : array of floats or []
		Array of extra features (each: dimension extraSize).
	ds : ExternalData or DataSource
		Data structure that contains more detailed information about the underlying data (for example finer curves for plotting).
	problemName : string
		name of the problem (for identification purposes and the output sub folder)
	numTens : int
		Number of generalized structural tensors to use (at least 1).
	numDir : int
		Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).
	batchSize : int
		Batch size during training.
	epochs : int
		Number opochs to use during training.
	incomp : boolean
		Switch to determine whether the material should be considered as incompressible or not.
		
	"""
	
	if problem==0:
		problemName = 'Treloar'
		
		ds = Treloar()
		ds.createPlotCurves()
		F, P11 = ds.createTrainingCurves()
		extra = []
	elif problem==1:	
		problemName = 'GeneralizedMooneyRivlin'

		Psi = CM.MooneyRivlin6term_wrapper(1.6E-1, -1.4E-3, 3.9E-5, 1.5E-2, -2E-6, 1E-10)
		
		ds = StrainEnergy(Psi, 7, 4, 5.5, True)
		ds.createPlotCurves()
		F, P11 = ds.createTrainingCurves()
		extra = []
	elif problem==2:
		problemName = 'GeneralizedMooneyRivlinNoise'

		Psi = CM.MooneyRivlin6term_wrapper(1.6E-1, -1.4E-3, 3.9E-5, 1.5E-2, -2E-6, 1E-10)
		
		ds = StrainEnergy(Psi, 7, 4, 5.5, True)
		ds.createPlotCurves()
		ds.createTrainingCurves(numP_ut=50, numP_bt=50, numP_ps=50)
		mu, sigma = 0, 0.05
		ds.noiseTrainingData(mu, sigma)
		F, P11 = ds.getTrainingData()
		extra = []
	
	
	numTens = 1
	numDir = 0
	batchSize = 4
	epochs = 4000

	incomp = True

	return F, P11, extra, ds, problemName, numTens, numDir, batchSize, epochs, incomp


######################
######################
### CLASS: EXTERNAL DATA
######################
######################

class ExternalData:
	# Needs to be adjusted to specific input file
	def __init__(self, file):
		self.inputs = np.loadtxt(file, delimiter=',', skiprows=1)
	
	def getData(self):
		self.extra = self.inputs[:,:7]
		self.P11 = self.inputs[:,8]
		lam = self.inputs[:,7] + 1.
		
		self.F = CM.defGrad_ut(lam)
		return self.F, self.P11, self.extra
		
######################
######################
### CLASS: CURVE
######################
######################

class curve:
	def __init__(self, lam, P11, F):
		self.lam = lam
		self.P11 = P11
		self.F = F
		self.P11_pred = np.zeros(P11.shape)
		self.P11_pred[:] = np.NaN # Initialize a spot where later on the prediction can be stored
		
	def predict(self, model): 
		self.P11_pred = model.predict(self.F) # Predict stress using the provided keras model
		
	def noise(self, mu, sigma): # Add noise to curve
		noise = np.random.normal(mu, sigma, self.P11.shape)
		self.P11 = np.multiply(self.P11, 1.0+noise)
		
######################
######################
### CLASS: DATA SOURCE
######################
######################

class DataSource:
	def createPlotCurves(self, numP_ut=100, numP_bt=100, numP_ps=100): # Create plot curves (fine for nicer visuals)
		self.plot_ut, self.plot_bt, self.plot_ps = self.createMainCurves(numP_ut, numP_bt, numP_ps)

	def createTrainingCurves(self, numP_ut=15, numP_bt=15, numP_ps=15): # Create training curves (course)
		self.train_ut, self.train_bt, self.train_ps = self.createMainCurves(numP_ut, numP_bt, numP_ps)
		
		return self.getTrainingData()
	
	def getTrainingData(self): # Combine all three main load cases to obtain one array of training data
		P11 = np.concatenate([self.train_ut.P11, self.train_bt.P11, self.train_ps.P11])
		F   = np.concatenate([self.train_ut.F  , self.train_bt.F  , self.train_ps.F  ])
		
		return F, P11
		
	def predict(self, model): # predict all values based on the keras model provided
		self.train_ut.predict(model)
		self.train_bt.predict(model)
		self.train_ps.predict(model)
		self.plot_ut.predict(model)
		self.plot_bt.predict(model)
		self.plot_ps.predict(model)

	def noiseTrainingData(self, mu, sigma):
		self.train_ut.noise(mu, sigma)
		self.train_bt.noise(mu, sigma)
		self.train_ps.noise(mu, sigma)

		
######################
######################
### CLASS: STRAINENERGY
######################
######################

class StrainEnergy(DataSource):
	def __init__(self, Psi, lamMax_ut, lamMax_bt, lamMax_ps, incomp): # Create a CANN but with the analytical strain energy function provided in Psi. Now the CANN is technically not a ANN but only a computational graph that can be executed to obtain reference values. Using this approach the entire continuum mechanics frame work build into the CANNs can be utilized. 
		self.Psi = Psi
		self.lamMax_ut = lamMax_ut
		self.lamMax_bt = lamMax_bt
		self.lamMax_ps = lamMax_ps
		self.incomp = incomp

		CANN_mod = CANN.CANN_wrapper(Psi, subANNs.dir_layers_wrapper(), subANNs.w_layers_wrapper(), numTens=1, numDir=0, numExtra=0, incomp=True, computePsiOnly=True)

		self.model_fit, self.model_full = CANN_mod() # Create a CANN model with analytical strain energy (instead of an ANN)
	
	def createMainCurves(self, numP_ut, numP_bt, numP_ps): # Generate the load curves for the three different load cases
		ut = self.createCurve(self.lamMax_ut, numP_ut, CM.defGrad_ut)
		bt = self.createCurve(self.lamMax_bt, numP_bt, CM.defGrad_bt)
		ps = self.createCurve(self.lamMax_ps, numP_ps, CM.defGrad_ps)
		
		return ut, bt, ps

	def createCurve(self, lamMax, numP, defGrad): # Create one particular load curve
		lam = np.linspace(1, lamMax, numP)
		F = defGrad(lam)
		P11 = self.model_fit.predict(F, batch_size=100000) # Once we ran into an issue with the batch size. While this issue may not persist any more due to a variety of changes in the code since, it was never investigated further. Hence, a huge batch size is put to avoid potential problems. ¯\_(ツ)_/¯

		return curve(lam, P11, F)

		
######################
######################
### CLASS: TABULAR
######################
######################

class Tabular(DataSource):
	def createMainCurves(self, numP_ut, numP_bt, numP_ps): # Generates the three main curves based on the base data provided in it's child class
		ut = Tabular.interpolateCurve(self.base_ut, numP_ut, CM.defGrad_ut)
		bt = Tabular.interpolateCurve(self.base_bt, numP_bt, CM.defGrad_bt)
		ps = Tabular.interpolateCurve(self.base_ps, numP_ps, CM.defGrad_ps)

		return ut, bt, ps
		
	def interpolateCurve(data, numP, defGrad): # Interpolates given data points to defined number of equi-distant points
		lam_tmp = data[:,0]
		P11_tmp = data[:,1]

		lam = np.linspace(np.amin(lam_tmp), np.amax(lam_tmp), numP)
		P11 = np.interp(lam, lam_tmp, P11_tmp)
		F = defGrad(lam)
		
		return curve(lam, P11, F)

######################
######################
### CLASS: TRELOAR
######################
######################

class Treloar(Tabular):
	def __init__(self):
		self.base_ut = np.array([
			[1.00000000000000e+000, 0.00000000000000e+000],
			[1.02423152035831e+000, 2.82511465836337e-002],
			[1.12114496145751e+000, 1.31851324390999e-001],
			[1.23416218812609e+000, 2.16667965812110e-001],
			[1.39557925384108e+000, 3.10964857764641e-001],
			[1.58118991891226e+000, 4.05299670719298e-001],
			[1.90388500666778e+000, 4.90444960825498e-001],
			[2.17820553604497e+000, 5.84918817454616e-001],
			[2.42025529261320e+000, 6.69937704220397e-001],
			[3.01728355007995e+000, 8.58961259482884e-001],
			[3.57398914195306e+000, 1.04792161307516e+000],
			[4.03390769606738e+000, 1.22732587413177e+000],
			[4.76019600944654e+000, 1.58583102822797e+000],
			[5.36552896637882e+000, 1.95355098584072e+000],
			[5.75311952910542e+000, 2.32092965443433e+000],
			[6.14876198461671e+000, 2.67891655483482e+000],
			[6.40730737710962e+000, 3.03668856955660e+000],
			[6.60136178532078e+000, 3.41316827866038e+000],
			[6.85991981814773e+000, 3.78034470190932e+000],
			[7.05392366502272e+000, 4.11920677690442e+000],
			[7.15120367580914e+000, 4.49553480199970e+000],
			[7.25654821971429e+000, 4.87187546742902e+000],
			[7.40217750821095e+000, 5.22006610894704e+000],
			[7.49945751899737e+000, 5.59639413404232e+000],
			[7.60526975526207e+000, 6.32069791497690e+000]
			])
		self.base_bt = np.array([
			[1.00000000000000e+000, 0.00000000000000e+000],
			[1.06722689075630e+000, 1.64113785557987e-001],
			[1.11764705882353e+000, 2.56017505470460e-001],
			[1.20168067226891e+000, 3.54485776805252e-001],
			[1.30252100840336e+000, 4.59518599562363e-001],
			[1.42016806722689e+000, 5.38293216630197e-001],
			[1.68067226890756e+000, 6.76148796498906e-001],
			[1.94117647058824e+000, 7.94310722100656e-001],
			[2.48739495798319e+000, 9.97811816192560e-001],
			[3.03361344537815e+000, 1.28665207877462e+000],
			[3.43697478991597e+000, 1.49671772428884e+000],
			[3.75630252100840e+000, 1.77899343544858e+000],
			[4.06722689075630e+000, 2.04814004376368e+000],
			[4.44537815126050e+000, 2.51422319474836e+000]
			])
		self.base_ps = np.array([
			[1.000000000000000, 0.000000000000000],
			[1.042660186857600, 0.065577307396771],
			[1.120976649297660, 0.162546632916218],
			[1.192198240575910, 0.245264158911253],
			[1.306365892641350, 0.330910881661187],
			[1.442013072235140, 0.416598382092164],
			[1.849331444832830, 0.591079282993533],
			[2.385552971139730, 0.760109567195533],
			[2.964707563964290, 0.934916689545244],
			[3.457905063866830, 1.118103625349220],
			[3.936860844367630, 1.284177527675630],
			[4.344153228426260, 1.464353711362620],
			[4.665618462258140, 1.624433294575760],
			[4.929739984666760, 1.798642344270180]
			])
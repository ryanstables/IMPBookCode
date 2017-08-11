# Python Class to read and process the SAFE (http://www.semanticaudio.co.uk) Data. 
# - The relative flie structures and csv naming conventions need to be preserved
# - class dependencies are: 
	# NLTK (http://www.nltk.org), 
	# SKLearn (http://scikit-learn.org/stable/), 
	# SciPy/NumPy/Matplotlib

import scipy.signal as sp
import os
import numpy as np
import csv 
import re
import nltk as nt
from matplotlib.mlab import PCA
import itertools
import string

#Todo: 
	# main thing: change structure of classes so the descriptor and context 
	# objects inherit all of the functions and attributes from the SAFEData class
	# e.g. have instances, are searchable and can be split. 

# Descriptor:
	# think of a better way to do the split-terms-by-context thing! 
	# split dataset by effect and apply sorting functions
	# the remove jibberish function doesn't seem to apply to the loadTerms fn?
	# filter out jibberish terms (a, for, test, etc) and improve the stemming function 
	# do sort by x functions (using the SAFEData.uniqueTerms() to loop through and sort the instances)
	# calculate generality across other context distributions (not just effects)
	# assign searchTerm member to descriptor class
	# improve the stemming functions 
	# make the search-term in the descriptor a dictionary with the contextType as the keys
# MYSQL: 
	# - set cron job/check for updates (Raf)
	# - add .update function to SAFEData class

# Class for a plug-in type...
class SAFEPlugin(object):
	def __init__(self, IDNum):
		self.plugInName  = []
		self.numParams   = []
		self.params 	 = []
		self.pIdx 		 = []
		self.mIdx 		 = []
		self._setMembers(IDNum)

	def _setMembers(self, IDNum):
		plugInNames = ['Compressor', 'Distortion', 'EQ', 'Reverb']
		paramNames = [
						['Threshold', 'Ratio', 'Knee', 'AttackTime', 'ReleaseTime', 'MakeUpGain'],
						['InputGain', 'Knee', 'Bias', 'Tone', 'OutputGain'],
						['Band1Gain', 'Band1Frequency', 'Band2Gain', 'Band2Frequency', 'Band2QFactor', 'Band3Gain', 'Band3Frequency', 'Band3QFactor', 'Band4Gain', 'Band4Frequency', 'Band4QFactor', 'Band5Gain', 'Band5Frequency'],
						['DampingFrequency', 'Density', 'BandwidthFrequency', 'Decay', 'PreDelay', 'Size', 'Gain', 'Mix', 'EarlyMix']
						]
		numParams 	= [len(i) for i in paramNames]	
		paramIdxs = [[5, 10], [5, 9], [5, 17], [5, 13]]
		mdIdxs    = [[11, 17],[10, 16],[18, 24],[14, 20]]	
		
		self.plugInName  = plugInNames[IDNum]
		self.numParams   = numParams[IDNum]
		self.params 	 = paramNames[IDNum]
		self.pIdx 		 = paramIdxs[IDNum]
		self.mIdx 		 = mdIdxs[IDNum]

# Class for an entry into the dataset...
class SAFEInstance(object):
	def __init__(self):		
		#checks
		self.hasFeatures = False
		self.hasUserData = False
		# User Data ....
		self.ID		 	 	= []
		self.rawTermString  = []
		self.terms    		= [] 
		self.numTerms 		= []
		self.ip 			= []
		self.IO 			= []
		# Parameter settings....		
		self.effectType 	= []
		self.numParams 		= []
		self.params   		= []
		# Additional Metadata...
		self.genre  		= []
		self.instrument 	= []
		self.location		= []
		self.experience 	= []
		self.age 			= []
		self.language 		= []
		# Audio features........
		self.featuresBefore = []
		self.featuresAfter  = []
		self.features  		= []

	def setUserData(self, userDataRow, plugID):
		# User Data ....
		self.ID		 	 	= userDataRow[0]
		self.rawTermString  = userDataRow[1]
		self.terms    		= [] 
		self._explodeAndPreprocess(self.rawTermString)
		self.numTerms 		= len(self.terms)
		self.ip 			= userDataRow[2]
		self.IO 			= [userDataRow[3], userDataRow[4]]
		# Parameter settings....		
		plugin = SAFEPlugin(plugID)
		self.effectType.append(plugin.plugInName)
		self.numParams = plugin.numParams
		self.params   		= map(float, userDataRow[plugin.pIdx[0]:plugin.pIdx[1]+1])
		# Additional Metadata...
		self.genre  		= self._cleanMetadata(userDataRow[plugin.mIdx[0]], mdType='Genre')
		self.instrument 	= self._cleanMetadata(userDataRow[plugin.mIdx[0]+1], mdType='Instr')
		self.location		= self._cleanMetadata(userDataRow[plugin.mIdx[0]+2], mdType='Loc')
		self.experience 	= self._cleanMetadata(userDataRow[plugin.mIdx[0]+3], mdType='Exp')
		self.age 			= self._cleanMetadata(userDataRow[plugin.mIdx[0]+4], mdType='Age')
		self.language 		= self._cleanMetadata(userDataRow[plugin.mIdx[0]+5], mdType='Lang')

	def _explodeAndPreprocess(self, inputString='*'):
		# remove dodgy chars...
		printable = set(string.printable)
		inputString = filter(lambda x: x in printable, inputString)
		inputString.encode('ascii',errors='ignore')

		# expand multiple into an array...
		termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
		termList = [re.sub('[^a-zA-Z]', '', i) for i in termList]
		termList = [i for i in termList if i]
				
		self.numTerms = len(termList)
		# preprocess (make lower case and remove non-alpha)
		termList = [x.lower() for x in termList]
		# apply stemming...
		stemmed = []
		for i in termList:
			try:
				stemmed.append(ps.stem(i).encode('UTF8')) #encoding causes some problems when comparing strings?!
			except:
				stemmed.append(i)
		# Additional stemming conditions.......................		
		stemmed = [a.replace(a, a[:-1]) if a[-1] in ('i', 'y') and len(a) > 3 else a for a in stemmed]
		stemmed = [a.replace(a, a[:-2]) if a[-2:] == 'ly' else a for a in stemmed]
		stemmed = [a.replace(a, a[:-2]) if a[-2:] in ('er', 'ed') and a[-3] not in ('h', 'v', 't') else a for a in stemmed]
		stemmed = [a.replace(a, a[:-3]) if a[-3:] == 'ing' else a for a in stemmed]
		stemmed = [a.replace(a, a[:-2]) if (a[-2:] == 'dy' and a[-3] == 'd') else a for a in stemmed]		
		stemmed = [a.replace(a, a[:-2]) if (a[-2:] == 'th' and a[-3] not in ('a', 'e', 'i', 'o', 'u', 'd')) else a for a in stemmed]
		stemmed = [a.replace(a, '') if len(a) < 3 else a for a in stemmed]
		stemmed = [a.replace(a, 'warm') if a == 'war' else a for a in stemmed]
		stemmed = [a.replace(a, 'beautiful') if a == 'beauts' else a for a in stemmed]
		stemmed = [a.replace(a, 'mud') if a == 'mudd' else a for a in stemmed]
		stemmed = [a.replace(a, 'bass') if a == 'bas' else a for a in stemmed]
		stemmed = [a.replace(a, 'tin') if a == 'tinn' else a for a in stemmed]
		self.hasUserData = True
		self.terms = stemmed

	def _cleanMetadata(self, inputString, mdType):
		printable = set(string.printable)
		if(mdType == 'Genre'):
		# remove dodgy chars...
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			else:
				# do some corrections...
				termList = ['folk' if i=='folklore' else i for i in termList]
				termList = ['reggae' if i in ('raggae', 'raggabreaks') else i for i in termList]
				termList = ['lofi' if i in ('lo', 'fi') else i for i in termList]
				termList = ['lofi' if i in ('cumbia', 'simple') else i for i in termList]
				termList = ['electronica' if i in ('industrial/electronic', 'electronic', 'industrial', 'glitch') else i for i in termList]
				termList = ['experimental' if i in ('fixtotalcrapacquisition', 'hat') else i for i in termList]
				termList = ['' if i in ('01', '02', '04', '05', '07', '08', '09', '1', '10', '11', '12', '13', '14', '15',
 '16', '17', '170', '18', '19', '20', '22', '23', '24', '26', '29', 'a', 'fix', 'total', 'crap', 'acquisition') else i for i in termList]
			return termList
		elif(mdType == 'Instr'):
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			else:
				# do some corrections...
				termList = ['vocals' if i in ('voice', 'throat', 'singing', 'male', 'males', 'female', 'bvox', 'bvs','bg', 'backing', 'vox', 'vocal', 'backingvox', 'leadvocals', 'leadvocal', 'lead', 'leadvox', 'local', 'main') else i for i in termList]
				termList = ['guitar' if i in ('tanglewood', 'telecaster', 'pu', 'gtr', 'guitars', 'guitat', 'git', 'gitar', 'electric', 'elerctric', 'e', 'electricguitar', 'acousticguitar', 'acoustic') else i for i in termList]
				termList =['bass' if i in ('base', 'basss') else i for i in termList]
				termList =['drums' if i in ('chanel', 'aux', 'drum', 'clicks', 'kit', 'delaybus', 'oberhead', 'of', 'oh', 'overhead', 'overheads') else i for i in termList]
				termList =['master' if i in ('daw', 'bus', 'group', 'mix', 'front', 'full' ) else i for i in termList]
				termList =['hihat' if i in ('hat', 'hats', 'hi', 'high', 'highhat') else i for i in termList]
				termList =['kick' if i in ('kcik', 'kik') else i for i in termList]
				termList =['keyboard' if i in ('key', 'keys', 'synth') else i for i in termList]
				termList =['saxophone' if i in ('sav' 'sax') else i for i in termList]
				termList =['snare' if i in ('snaro', 'snares') else i for i in termList]
				termList = ['' if i in ('12' '2' '3' '5' 'a') else i for i in termList]
			return termList			
		elif(mdType == 'Loc'):
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [re.sub('[^a-zA-Z]', '', i) for i in termList]	
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			else:
				termList = ['italy' if i in ('studiosza', 'vonte', 'za') else i for i in termList]
				termList = ['russia' if i=='moscow' else i for i in termList]
				termList = ['taiwan' if i=='taipei' else i for i in termList]
				termList = ['canada' if i=='montreal' else i for i in termList]
				termList = ['germany' if i in ('hamburg', 'hasselt') else i for i in termList]
				termList = ['usa' if i in ('blues', 'how', 'room', 'descktop', 'home') else i for i in termList]
				termList = ['spain' if i in ('larioja', 'la', 'rioja') else i for i in termList]
				termList = ['' if i in ('shit', 'simple', 'a', 'cumbia') else i for i in termList]
				termList = ['africa' if i in ('democraticrepublicofthecongo', 'democratic', 'republic', 'of', 'congo', 'bophutabitswaba') else i for i in termList]
				termList = ['uk' if i in ('great', 'britain', 'birminham', 'birmingham', 'birminghamuk', 'brimingham', 'birminhamuk', 'london', 'bcu', 'unitedkingdom','england', 'greatbritain', 'dmtoffice', 'dmt', 'office', 'united', 'kingdom') else i for i in termList]
			return termList
		elif(mdType == 'Exp'):
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			return termList
		elif(mdType == 'Age'):				
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			return termList
		elif(mdType == 'Lang'):
			inputString = filter(lambda x: x in printable, inputString)
			inputString.encode('ascii',errors='ignore')
			# expand multiple into an array...
			termList = filter(None, re.split('[^0-9a-zA-Z]+', inputString))
			termList = [re.sub('[^a-zA-Z]', '', i) for i in termList]	
			termList = [i.lower() for i in termList if i]		
			if not termList:
				termList = ['']
			else:
				termList = ['english' if i in ('simple', 'a', 'cumbia') else i for i in termList]        
				termList = ['spanish' if i in ('esp', 'espanya') else i for i in termList]      
			return termList
		else:
			return inputString

	def setFeatureData(self, featureRow):		
		# remove empty values... (remove NULL/NaN too?)	
		features = [i if i else 0 for i in featureRow[2:]]		
		if(featureRow[1] == 'processed'):
			# self.featuresAfter = features
			# print 'p', featureRow[0]
			self.featuresAfter = map(float, features)
		elif(featureRow[1] == 'unprocessed'):
			# self.featuresBefore = features
			# print 'u', featureRow[0]
			self.featuresBefore = map(float, features)
		if(len(self.featuresAfter) and len(self.featuresBefore)):
				self.features = list(np.array(self.featuresAfter) - np.array(self.featuresBefore))
		self.hasFeatures = True
		
# Class for capturing groups of instances...
class SAFEDescriptor(SAFEInstance):
	def __init__(self, searchTerms=''):
		# instances...
		self.terms 			= []
		self.numTerms 		= len(self.terms)
		self.IPs 			= []
		self.instances 		= []
		self.numInstances 	= len(self.instances)
		# metrics...
		self.confidence 	= []
		self.popularity		= []
		self.numEffects 	= []
		self.generality 	= []
		self.plugInDistribution =[]
		self.metadataDistribution = []
		self.termsPerUser = []
		self.normalise = True
		self.searchTerms = searchTerms
		self.unique = {}

	def insert(self, inst):
		self.instances.append(inst)
		self.terms.append(inst.terms)
		self.IPs.append(inst.ip)
		self.numTerms 		= len(self.terms)
		self.numInstances 	= len(self.instances)

	def getFeatureMatrix(self, normalise=False, range=[0,1], featureType='diff'):
		self.normalise=normalise				
		if(featureType=='unprocessed' or featureType=='before'):
			x = np.vstack([instance.featuresBefore for instance in self.instances])
		elif(featureType=='processed' or featureType=='after'):
			x = np.vstack([instance.featuresAfter for instance in self.instances])
		elif(featureType=='diff'):
			x = np.vstack([instance.features for instance in self.instances])
		else: 
			x = np.vstack([instance.features for instance in self.instances])
		if self.normalise:
			x = np.vstack([self._normalise(row, range[0], range[1]) for row in x.transpose()]).transpose()			
		return x

	def getUnique(self):	
		if self.instances:	
			self.unique={
			'terms': np.unique(list(itertools.chain(*[i.terms for i in self.instances]))),
			'effectType': np.unique(list(itertools.chain(*[i.effectType for i in self.instances]))),
			'genre': np.unique(list(itertools.chain(*[i.genre for i in self.instances]))),
			'instrument':  np.unique(list(itertools.chain(*[i.instrument for i in self.instances]))),
			'location':  np.unique(list(itertools.chain(*[i.location for i in self.instances]))),
			'experience':  np.unique(list(itertools.chain(*[i.experience for i in self.instances]))),
			'age':  np.unique(list(itertools.chain(*[i.age for i in self.instances]))),
			'language':  np.unique(list(itertools.chain(*[i.language for i in self.instances])))
			}
		return self.unique


	def getParameters(self):
		return [instance.params for instance in self.instances]

	def getUniqueTerms(self):
		termList = np.unique(list(itertools.chain(*self.terms)))
		return [str(t) for t in termList]

	def getUniqueIPs(self):
		return np.unique(self.IPs)		

	def getNumTerms(self):
		return len(self.getUniqueTerms())

	def getConfidence(self, weightScores=False):		
		N = self.numInstances
		if N:
			features = self.getFeatureMatrix(normalise=self.normalise)
			self.confidence = np.mean(np.var(features, axis=0))
			# prevent divide-by-0...
			if self.confidence:	
				self.popularity = np.log(N)/self.confidence
			else:
				self.popularity = 0;
			# return based on inputFlag...
			if weightScores:				
				return self.popularity				
			else:				
				return self.confidence

	def getTermsPerUser(self):
		if len(self.getUniqueIPs()):
			self.termsPerUser = float(self.numInstances)/len(self.getUniqueIPs())
		else:
			self.termsPerUser = 0
		return self.termsPerUser


	def getPlugInDistribution(self, normalise=False, totalInstances=[1,1,1,1]):
		plugInNames = ['Compressor', 'Distortion', 'EQ', 'Reverb']
		x = [len([1 for i in self.instances if i.effectType[0] == plugin]) for plugin in plugInNames]
		if normalise:
			# if normalise flag is set, find the proportions of total terms...
			x = [float(x[i])/totalInstances[i] if totalInstances[i] else 0 for i in range(len(plugInNames))]		
		self.plugInDistribution = [i/float(np.sum(x)) for i in x]
		return self.plugInDistribution

	def getGenerality(self, normalise=True, totalInstances=[1, 1, 1, 1]):
		# use genre and instrumnt distribution too...
		if not sum(self.plugInDistribution):
			self.getPlugInDistribution(normalise=normalise, totalInstances=totalInstances)
		xAxis = np.array(range(len(self.plugInDistribution)))
		yAxis = np.array(sorted(self.plugInDistribution, reverse=True))
		
		if np.mean(xAxis):
			self.generality = np.sum(xAxis*yAxis)/np.mean(xAxis)
		else:
			self.generality = 0
		return self.generality

	def _normalise(self, x, nmin, nmax):
		den = np.max(x)-np.min(x)
		if den:
			return [(((float(i)-np.min(x))*(nmax-nmin))/den)+nmin for i in x]
		else:
			return np.zeros(len(x))

# Class for processing the full dataset...
class SAFEData(object):
	def __init__(self, folder, removeJibberish=False):
		# general stuff...
		self.plugInNames = ['Compressor', 'Distortion', 'EQ', 'Reverb']
		self.metadataClasses = ['Genre', 'Instrument', 'Location', 'Experience', 'Age', 'Language'] 
		self.states = ['unprocessed', 'processed', 'diff']
		self.dir = folder
		self.d = os.listdir(self.dir)
		# get stuff from the user data files...
		self.instances = []
		self._setUserData()
		# calculate instances and IDs...
		self.numInstances = len(self.instances)
		self.numInstancesExpanded = np.sum([len(i.terms) for i in self.instances])
		self.IDs = [self.instances[instance].ID for instance in range(self.numInstances)]
		# get the features from the AudioFeatureData files...
		self._setFeatureData()
		# calculate instances and clean up missing data...
		self.numInstancesWithUserData = np.sum([instance.hasUserData for instance in self.instances])
		self.numInstancesWithFeatureData = np.sum([instance.hasFeatures for instance in self.instances])
		self._removeEntriesWithMissingData(removeJibberish=removeJibberish)


		self.uniqueTerms=[]  #remove this and replace with the unique dict!!!
		self._getTerms()	#remove this and replace with the _getUnique() fn!!!

		self.unique = {} # this is a dictionary populated by the _getUnique() function.
		self._getUnique()

		self.descriptors=[]
		self.totalInstancesPerPlugin = [len([1 for i in self.instances if i.effectType[0] == plugin]) for plugin in self.plugInNames]

	def _getUnique(self):	
		if self.instances:	
			self.unique={
			'terms': np.unique(list(itertools.chain(*[i.terms for i in self.instances]))),
			'effectType': np.unique(list(itertools.chain(*[i.effectType for i in self.instances]))),
			'genre': np.unique(list(itertools.chain(*[i.genre for i in self.instances]))),
			'instrument':  np.unique(list(itertools.chain(*[i.instrument for i in self.instances]))),
			'location':  np.unique(list(itertools.chain(*[i.location for i in self.instances]))),
			'experience':  np.unique(list(itertools.chain(*[i.experience for i in self.instances]))),
			'age':  np.unique(list(itertools.chain(*[i.age for i in self.instances]))),
			'language':  np.unique(list(itertools.chain(*[i.language for i in self.instances])))
			}

	def toggleControlData(self, preserve=False):
		# This function removes the test data that wwas collected during an experiment on Warm/Bright terms at BCU.
		if self.numInstances:
			if preserve:
				self.instances = [self.instances[i] for i in range(len(self.instances)) if re.search('(jazz[0-9]|metal[0-9])', self.instances[i].genre[0]) or re.search('(jazz[0-9]|metal[0-9])', self.instances[i].instrument[0])]
			else:
				self.instances = [self.instances[i] for i in range(len(self.instances)) if not re.search('(jazz[0-9]|metal[0-9])', self.instances[i].genre[0]) and not re.search('(jazz[0-9]|metal[0-9])', self.instances[i].instrument[0])]
			self.numInstances = len(self.instances)
			self.numInstancesExpanded = np.sum([len(i.terms) for i in self.instances])
			self.IDs = [self.instances[instance].ID for instance in range(self.numInstances)]
			self._getTerms()
			self.totalInstancesPerPlugin = [len([1 for i in self.instances if i.effectType[0] == plugin]) for plugin in self.plugInNames]
			self._getUnique() #recalculate the unique terms
		else:
			print 'No instances have ben loaded yet!!'


	def _setUserData(self):
		instances = []
		# read the user data.....
		for i in range(len(self.d)):
			if(self.d[i]=="SAFECompressorUserData.csv"):
				compressorFeatures = []
				with open(self.dir+self.d[i], 'rb') as data:
					CompressorData = np.array(list(csv.reader(data)))
					for j in range(len(CompressorData)):
						inst = SAFEInstance()
						inst.setUserData(CompressorData[j], 0)
						instances.append(inst)

			elif(self.d[i]=="SAFEDistortionUserData.csv"):
				distortionFeatures = []
				with open(self.dir+self.d[i], 'rb') as data:
					DistortionData = np.array(list(csv.reader(data)))
					for j in range(len(DistortionData)):						
						inst = SAFEInstance()
						inst.setUserData(DistortionData[j], 1)
						instances.append(inst)

			elif(self.d[i]=="SAFEEqualiserUserData.csv"):
				EQFeatures = []
				with open(self.dir+self.d[i], 'rb') as data:
					EQData = np.array(list(csv.reader(data)))
					for j in range(len(EQData)):						
						inst = SAFEInstance()
						inst.setUserData(EQData[j], 2)
						instances.append(inst)

			elif(self.d[i]=="SAFEReverbUserData.csv"):
				reverbFeatures = []
				with open(self.dir+self.d[i], 'rb') as data:
					ReverbData = np.array(list(csv.reader(data)))					
					for j in range(len(ReverbData)):						
						inst = SAFEInstance()
						inst.setUserData(ReverbData[j], 3)
						instances.append(inst)
		self.instances = instances			
			
	def _setFeatureData(self):
		for i in self.d:
		    if(i=="SAFECompressorAudioFeatureData.csv"):
		    	# print "Processing: ", i
		        currentEffect = self.plugInNames[0]
		        with open(self.dir+i, 'rb') as data:
		            featureData = np.array(list(csv.reader(data)))
		        for featureEntry in featureData:
		            entryID = featureEntry[0]
		            entryState = featureEntry[1]  
		            for i in range(self.numInstances):
		                if entryID == self.IDs[i] and currentEffect == self.instances[i].effectType[0]:
		                        self.instances[i].setFeatureData(featureEntry)
		                        break
		    if(i=="SAFEDistortionAudioFeatureData.csv"):
		    	# print "Processing: ", i
		        currentEffect = self.plugInNames[1]
		        with open(self.dir+i, 'rb') as data:
		            featureData = np.array(list(csv.reader(data)))
		        for featureEntry in featureData:
		            entryID = featureEntry[0]
		            entryState = featureEntry[1]  
		            for i in range(self.numInstances):
		                if entryID == self.IDs[i] and currentEffect == self.instances[i].effectType[0]:
		                        self.instances[i].setFeatureData(featureEntry)
		                        break
		    if(i=="SAFEEqualiserAudioFeatureData.csv"):
		    	# print "Processing: ", i
		        currentEffect = self.plugInNames[2]
		        with open(self.dir+i, 'rb') as data:
		            featureData = np.array(list(csv.reader(data)))
		        for featureEntry in featureData:
		            entryID = featureEntry[0]
		            entryState = featureEntry[1]  
		            for i in range(self.numInstances):
		                if entryID == self.IDs[i] and currentEffect == self.instances[i].effectType[0]:
		                        self.instances[i].setFeatureData(featureEntry)
		                        break    
		    if(i=="SAFEReverbAudioFeatureData.csv"):
		    	# print "Processing: ", i
		        currentEffect = self.plugInNames[3]
		        with open(self.dir+i, 'rb') as data:
		            featureData = np.array(list(csv.reader(data)))
		        for featureEntry in featureData:		    
		            entryID = featureEntry[0]
		            entryState = featureEntry[1]  
		            for i in range(self.numInstances):
		                if entryID == self.IDs[i] and currentEffect == self.instances[i].effectType[0]:
		                        self.instances[i].setFeatureData(featureEntry)
		                        break
	
	def _removeEntriesWithMissingData(self, removeJibberish=False):
		# Todo: only remove the instance if there are no words in the entry!!!! 
		# Otherwise just remove that term from the list. 
		jibberishTerms = ['a', 'an', 'and', 'at', 'be', 'for', 'the', 'test']
		for i in reversed(range(self.numInstances)):
		    if(not self.instances[i].hasFeatures or not self.instances[i].hasUserData):
		        del self.instances[i]
		    if removeJibberish:
			    if(sum([sum([1 for jt in jibberishTerms if jt == t]) for t in self.instances[i].terms])):
			        del self.instances[i]



		# recalculate new IDs and lengths
		self.numInstances = len(self.instances)
		self.numInstancesExpanded = np.sum([len(i.terms) for i in self.instances])
		self.IDs = [self.instances[instance].ID for instance in range(self.numInstances)]
		self.numInstancesWithUserData = np.sum([instance.hasUserData for instance in self.instances])
		self.numInstancesWithFeatureData = np.sum([instance.hasFeatures for instance in self.instances])

	def _getTerms(self):
		# get a list of terms
		termList = [i.terms for i in self.instances]
		self.uniqueTerms = np.unique(list(itertools.chain(*termList)))

	def search(self, 
		term = '.*', 
		genre = '.*', 
		effectType = '.*', 
		instrument = '.*', 
		location = '.*', 
		experience = '.*', 
		age = '.*', 
		language = '.*', 
		matchExact = False, mute=False):

		# init a descriptor object and input the search term...
		descriptor = SAFEDescriptor(searchTerms=[value.lower() for key, value in locals().iteritems() if type(value) == str and value != '.*'])
		# init the regEx queries...
		terms_re, effect_re, genre_re, instr_re, loc_re, exp_re, age_re, lang_re = [[] for i in range(8)]

		# build the regex queries if they are needed. 
		# not building uneccessary queries speeds up the search process.
		# str() makes sure numpy strings are converted...
		if term != '.*': 
			terms_re = self._buildQuery(str(term), matchExact)
		if  effectType != '.*': 
			effect_re = self._buildQuery(str(effectType), matchExact)
		if genre != '.*': 
			genre_re = self._buildQuery(str(genre), matchExact)
		if instrument != '.*':
			instr_re = self._buildQuery(str(instrument), matchExact)
		if location != '.*': 
			loc_re = self._buildQuery(str(location), matchExact)
		if experience != '.*':
			exp_re = self._buildQuery(str(experience), matchExact)
		if age != '.*': 
			age_re = self._buildQuery(str(age), matchExact)
		if language != '.*':
			lang_re = self._buildQuery(str(language), matchExact)

		# loop over everything and check each instance for a match...
		# return 1 if the regex query didn't get built (this is quicker than searching through every field using .*)
		for instance in self.instances:	
			termResult = np.sum([np.count_nonzero([r.search(s) for s in instance.terms]) for r in terms_re]) if terms_re else 1
			effectResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.effectType]) for r in effect_re]) if effect_re else 1
			genreResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.genre]) for r in genre_re]) if genre_re else 1
			instResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.instrument]) for r in instr_re]) if instr_re else 1
			locationResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.location]) for r in loc_re]) if loc_re else 1
			experienceResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.experience]) for r in exp_re]) if exp_re else 1
			ageResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.age]) for r in age_re]) if age_re else 1
			languageResult  = np.sum([np.count_nonzero([r.search(s) for s in instance.language]) for r in lang_re]) if lang_re else 1
			if (termResult	and
				effectResult and
				genreResult and
				instResult and
				locationResult and
				experienceResult and
				ageResult and
				languageResult):
				descriptor.insert(instance)
		
		if descriptor.numInstances:
			if not mute:
				print  str(descriptor.numInstances), "instances found..."
			c   = descriptor.getConfidence()
			tpu = descriptor.getTermsPerUser()
			# this finds the proportion of the terms in the effect, rather than the absolute number of terms.
			pd  = descriptor.getGenerality(normalise=True, totalInstances=self.totalInstancesPerPlugin)
			return descriptor
		else:
			if not mute:
				print "No instances found...\n"
			return []		

	def _buildQuery(self, term, matchExact):
		# check the terms.............
		# if the term isn't an array, convert the string to an array...
		if type(term) is int or type(term) is float: 
			term = [str(term)]
		if type(term) is str: #not list?
		    term = [term]
		if(matchExact):
			if '.*' not in term:
				term = [r'\b'+t+r'\b' for t in term]
		# build the regex query....
		terms_re = [re.compile(t, re.IGNORECASE) for t in term]
		return terms_re

	def loadDescriptors(self, mute=False):
		if not mute:
			print "loading descriptors, this may take a while..."
		x = self.search(mute=mute)
		# this shouldn't happen, but if the terms didn't load, then find them again! 
		if not len(self.uniqueTerms):
			self.uniqueTerms = np.unique(list(itertools.chain(*[i.terms for i in x.instances])))
		for i in self.uniqueTerms:
			if not mute:
				print "[", i, "] ",
			self.descriptors.append(self.search(term=str(i), matchExact=True, mute=mute))
		return self.descriptors


	def splitByContext(self, context='effectType', mute=False):
		# check to make sure unique terms have been computed...
		if not self.unique:
			self._getUnique()

		# create an instance of a SAFEContext class...
		sc = SAFEContext(context, self.unique[context])

		if (context=='effectType'):
			for i in self.unique[context]:
				sc.insert(self.search(effectType=str(i), mute=mute))
		elif (context=='genre'):
			for i in self.unique[context]:
				sc.insert(self.search(genre=str(i), mute=mute))
		elif (context=='instrument'):
			for i in self.unique[context]:
				sc.insert(self.search(instrument=str(i), mute=mute))
		elif (context=='location'):
			for i in self.unique[context]:
				sc.insert(self.search(location=str(i), mute=mute))
		elif (context=='experience'):
			for i in self.unique[context]:
				sc.insert(self.search(experience=str(i), mute=mute))
		elif (context=='age'):
			for i in self.unique[context]:
				sc.insert(self.search(age=str(i), mute=mute))
		elif (context=='language'):
			for i in self.unique[context]:
				sc.insert(self.search(language=str(i), mute=mute))
		else:
			print 'incorrect context, no descriptors loaded!' 

		# finish calculating the other context attribtues...
		sc.getDistribution()
		sc.getUnique()
		sc.getFeatures()
		sc.getParameters()

		return sc

class SAFEContext(object):
	'''
		A class to hold an array of descriptors corresponding to fields from a given context...
	'''
	def __init__(self, context, uniqueTags):
		self.hasData = False
		self.context = context
		self.uniqueTags = uniqueTags
		self.contextDescriptors = []

		self.uniqueTermsPerField = []
		self.distribution =[]
		self.fieldNames = []		
		self.features = []
		self.parameters = []

		self.unique = []  # returns a N-D array where each element is a uniqeTags dictionary
		# run self.getUnique() to fill this!

	def insert(self, descriptor):
		# use this to load the instances into the class, do this per field in the context 
		self.contextDescriptors.append(descriptor)
		self.hasData = True 

	def getDistribution(self):
		if self.hasData:
			self.distribution =	[len(i.instances) for i in self.contextDescriptors]
			return self.distribution
		else: 			
			print 'load some data!'
			return []

	def getUnique(self):
		self.unique = [i.getUnique() for i in self.contextDescriptors]
		return self.unique

	def getFeatures(self, returnMeans=False, normalise=True, range=[0,1], featureType='diff'):
		self.features = [i.getFeatureMatrix(normalise=normalise, range=range, featureType=featureType) for i in self.contextDescriptors]
		
		# if returnMeans:
			# get the unique descriptor-wise means here. 
			# ideally, the individual descriptors would be derived, but this will take a while.
		
		return self.features

	def getParameters(self, returnMeans=False):
		self.parameters = [i.getParameters() for i in self.contextDescriptors]
	
		# if returnMeans:
			# get the unique descriptor-wise means here. 
			# ideally, the individual descriptors would be derived, but this will take a while.


	# def sort(self):
		# use this fn to re-order the descriptor objects based on 

# A class to find the frequency response (or to use) the SAFE  EQ parameters
class SAFEEQ(object):
    def __init__(self, params, fs=44100, fftSize=1024):
        # [0'Band1Gain', 1'Band1Frequency', 
        # 2'Band2Gain', 3'Band2Frequency', 4'Band2QFactor', 
        # 5'Band3Gain', 6'Band3Frequency', 7'Band3QFactor', 
        # 8'Band4Gain', 9'Band4Frequency', 10'Band4QFactor', 
        # 11'Band5Gain', 12'Band5Frequency'], 
        self.params = params
        self.fs = fs
        self.NFFT = fftSize
        self.freqResponse = []

    def biquad(self, f0, gain, Q, select):
        #get the filter coefficients using Biquad functions...
        if not Q: # avoid divide-by-zeros errors
            Q=1e-20;
        A = 10**(float(gain)/40.0);
        w0 = 2*np.pi*f0/float(self.fs);
        alpha = np.sin(w0)/(2*float(Q)); 
        a=[]
        b=[]
        if select == 0: # low shelf coefficients.
            b.append(A*( (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha))
            b.append(2*A*( (A-1) - (A+1)*np.cos(w0)))
            b.append(A*( (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha))
            a.append((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
            a.append(-2*( (A-1) + (A+1)*np.cos(w0)))
            a.append((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        elif select == 1: # peak coefficients
            b.append(1 + alpha*A)             
            b.append(-2*np.cos(w0))            
            b.append(1 - alpha*A)             
            a.append(1 + alpha/A)             
            a.append(-2*np.cos(w0))             
            a.append(1 - alpha/A)
        elif (select ==2): # high shelf coefficients
            b.append(A*( (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha))             
            b.append(-2*A*( (A-1) + (A+1)*np.cos(w0)))
            b.append(A*( (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha))             
            a.append((A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)             
            a.append(2*( (A-1) - (A+1)*np.cos(w0)))             
            a.append((A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        else: 
            a = 0
            b = 0
        return a, b


    def EQ(self, x):
        # apply the EQ to a signal...
        gain = [self.params[0], self.params[2], self.params[5], self.params[8],  self.params[11]];
        f0   = [self.params[1], self.params[3], self.params[6], self.params[9],  self.params[12]];
        Q    = [0.71,         self.params[4], self.params[7], self.params[10], 0.71];
        filterSelector = [0, 1, 1, 1, 2]
        y=x

        for f in range(len(filterSelector)):
            a,b = self.biquad(f0[f], gain[f], Q[f], filterSelector[f])   
            y = sp.lfilter(b, a, y)
        return y



    def getFreqResponse(self):
        # use the EQ function to get the magnitude response of the EQ...
        df = np.hstack([1, np.zeros(self.NFFT)])
        y = self.EQ(df)

        # do an FFT...
        Y = np.fft.fft(y, n=self.NFFT)
        t = np.linspace(0, self.fs, self.NFFT)
        yMag = np.absolute(Y)
        self.freqResponse = 20*np.log10(yMag[:(self.NFFT/2)])
        self.bins = t[:(self.NFFT/2)]

        # Add optional normalisation here...
        return self.bins, self.freqResponse

# A class to find the transfer function (or to use) the SAFE Distortion parameters
class SAFEDistortion(object):
    def __init__(self, params):
    	# ['InputGain', 'Knee', 'Bias', 'Tone', 'OutputGain']
		self.inputGain= 10**(params[0]/20.0)
		if params[1]:
			self.knee = params[1]
		else:
			self.knee = 10e-16
		self.bias = params[2]
		self.tone = params[3]
		self.outputGain = 10**(params[4]/20.0)

    def getTF(self):
		numSamples = 100;
		xIn = np.linspace(-1.0, 1.0, numSamples)
		xOut = np.linspace(-1.0, 1.0, numSamples) * self.inputGain

		# Auxiliary variables...
		oneMinusBias = 1.0 - self.bias
		oneMinusKnee = 1.0 - self.knee
		onePlusKnee  = 1.0 + self.knee

		# Knee parameters (calculate for speed)...
		c2 = - self.outputGain / (4.0 * self.knee)
		c1 = self.outputGain * onePlusKnee / (2.0 * self.knee)
		c0 = - self.outputGain * oneMinusKnee * oneMinusKnee / (4.0 * self.knee)

		# # main loop...
		for sample in range(numSamples):
			if(xOut[sample]  > oneMinusKnee):								
				if(xOut[sample] >= onePlusKnee):# positive clipping
					xOut[sample] = self.outputGain					
				else: # soft knee (positive)
					xOut[sample] = c2*xOut[sample]*xOut[sample] + c1*xOut[sample] + c0
			else:
				if(xOut[sample]  < -oneMinusBias*oneMinusKnee):
					if(xOut[sample] <= -oneMinusBias*onePlusKnee): # negative clipping
						xOut[sample] = -oneMinusBias*self.outputGain
					else: # soft knee (negative)
						xOut[sample] = -c2*xOut[sample]*xOut[sample]/oneMinusBias + c1*xOut[sample] - c0*oneMinusBias
  				else:
  					xOut[sample] = self.outputGain*xOut[sample]
		return xIn, xOut
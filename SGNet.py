
"""
Author: LU
Feb 2021
Stacked Generalization of CNN Ensembles.
Orthogonal Nets are instantiated with pretrained DeepUNET3D CNN models.
For DeepUNET3D architecture and training details, please refer to the following publication:
Umapathy, L., G. G. Perez-Carrillo, M. B. Keerthivasan, J. A. Rosado-Toro, M. I. Altbach, B. Winegar, 
C. Weinkauf, A. Bilgin, and Alzheimer’s Disease Neuroimaging Initiative. 
"A Stacked Generalization of 3D Orthogonal Deep Learning Convolutional Neural Networks for Improved Detection 
of White Matter Hyperintensities in 3D FLAIR Images." American Journal of Neuroradiology (2021).
DOI: https://doi.org/10.3174/ajnr.A6970

"""
import os, numpy as np
import time 

from Model_setup import *
from utils import *

class SGNet:
    '''
    StackGen-Net class with pretrained weights for Orthogonal CNNs and Meta CNN
    to predict WMH lesion masks for 3D FLAIR volumes (isotropic resolution)
    '''
    def __init__(
        self,
        visible_gpu='0',
        loadPath="Pretrained_models/",
    ):
        self.visible_gpu = visible_gpu
       
        self.opShape = (200,200)
        self.blkSize = 7
        self.class_weights = [0.5,120]
        self.loadPath = loadPath
        self.modelSavePath_ax = os.path.join(self.loadPath,'DeepUNET3D_pretrained_ax.h5')
        self.modelSavePath_sag = os.path.join(self.loadPath,'DeepUNET3D_pretrained_sag.h5')
        self.modelSavePath_cor = os.path.join(self.loadPath,'DeepUNET3D_pretrained_cor.h5')
        self.model_ax  = self.build_OrthogonalNet('Axial')
        self.model_sag = self.build_OrthogonalNet('Sagittal')
        self.model_cor = self.build_OrthogonalNet('Coronal')
  
    def build_OrthogonalNet(self, orientation):
        if orientation is 'Axial':
            model = loadSavedModel(self.modelSavePath_ax, self.class_weights)
        elif orientation is 'Sagittal':
            model = loadSavedModel(self.modelSavePath_sag, self.class_weights)
        elif orientation is 'Coronal':
            model = loadSavedModel(self.modelSavePath_cor, self.class_weights)
        else:
            print('Orientation not supported')
        return model
    
    def predictOrthogonalNets(self, ipImg):
        ipImg_A,origShape_A = reformat_inputOrientation(ipImg,'Axial', self.opShape)
        predImg_ax =  prediction_testVolume(ipImg_A,self.model_ax,self.blkSize)
        ipImg_S,origShape_S = reformat_inputOrientation(ipImg,'Sagittal',self.opShape)
        predImg_sag = prediction_testVolume(ipImg_S,self.model_sag,self.blkSize)
        ipImg_C,origShape_C = reformat_inputOrientation(ipImg,'Coronal',self.opShape)
        predImg_cor = prediction_testVolume(ipImg_C,self.model_cor,self.blkSize)   
        predImg_A = reformat_outputOrientation(predImg_ax,origShape_A,'Axial')
        predImg_S = reformat_outputOrientation(predImg_sag,origShape_S,'Sagittal')
        predImg_C = reformat_outputOrientation(predImg_cor,origShape_C,'Coronal')
        predImg_A = binarizePosterior(predImg_A,0.5)
        predImg_S = binarizePosterior(predImg_S,0.5)
        predImg_C = binarizePosterior(predImg_C,0.5)
        return predImg_A, predImg_S, predImg_C
    
    def predict_averageEnsemble(self,ipImg):
        tStart = time.time()
        predImg_A, predImg_S, predImg_C = self.predictOrthogonalNets(ipImg) 
        print('Averaging Orthogonal posteriors...')
        predictedMask = np.true_divide(predImg_A + predImg_S + predImg_C, 3)
        predictedMask = binarizePosterior(predictedMask)
        print("Time Elapsed : ",time.time() - tStart)
        return predictedMask
    
    def predict_MVEnsemble(self,ipImg):
        tStart = time.time()
        predImg_A, predImg_S, predImg_C = self.predictOrthogonalNets(ipImg) 
        print('Majority voting Orthogonal predictions...')
        temp_A = binarizePosterior(predImg_A)
        temp_S = binarizePosterior(predImg_S)
        temp_C = binarizePosterior(predImg_C)
        predictedMask = np.logical_and(temp_A,temp_S,temp_C)
        print("Time Elapsed : ",time.time() - tStart)
        return predictedMask
    
        
        
        
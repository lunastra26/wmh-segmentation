"""
Utilities for WMH segmentation using StackGen-Net (SGNet) 
Author: LU
Feb 2021
Umapathy, L., G. G. Perez-Carrillo, M. B. Keerthivasan, J. A. Rosado-Toro, M. I. Altbach, B. Winegar, 
C. Weinkauf, A. Bilgin, and Alzheimer’s Disease Neuroimaging Initiative. 
"A Stacked Generalization of 3D Orthogonal Deep Learning Convolutional Neural Networks for Improved Detection 
of White Matter Hyperintensities in 3D FLAIR Images." American Journal of Neuroradiology (2021).
DOI: https://doi.org/10.3174/ajnr.A6970
"""


# import files

import numpy as np
import nibabel as nib
from skimage.exposure import rescale_intensity as rescale
from skimage.restoration import denoise_tv_bregman as denoise_tv



'''
This function takes a 3D FLAIR image volume and a model with weights loaded into it
'''
def prediction_testVolume(ipImg,model,blockSize=7): 
    x_test = createTestVolumefromImg(ipImg,blockSize)   
    y_pred = model.predict(x_test)
    predImg = createImgfromTestVolume(y_pred,ipImg.shape,blockSize)
    return predImg


def createTestVolumefromImg(ipImg,blockSize=7):
    ''' Creates a 2.5D block for evaluation using trained Orthogonal Nets.   
    Blocks are created using a sliding window
    '''
    numSlices = ipImg.shape[2]
    tempNum = int(np.floor(blockSize/2))
    testVolume = []
    for idx in range(tempNum,numSlices - tempNum-1):
        slc2fetch1 = range(idx - tempNum,idx + tempNum + 1)
        x_temp = np.expand_dims(ipImg[:,:,slc2fetch1],axis=0)
        testVolume.append(x_temp)          
    testVolume = np.concatenate(testVolume,axis=0)
    testVolume = np.expand_dims(testVolume,axis=-1)  
    return testVolume

def createImgfromTestVolume(predImg,ipShape,blockSize): 
    '''  Converts 2.5D posteriors to posterior volume.   
        The center slice of each predicted block is retained.     
    '''
    final_prediction = np.zeros((ipShape))
    count=0
    numSlices = ipShape[2]
    tempNum = int(np.floor(blockSize/2))
    for idx in range(tempNum,numSlices - tempNum-1):
        final_prediction[:,:,idx] = predImg[count,:,:,tempNum,0]
        count+= 1
    return final_prediction

def load3DFLAIR(nii_fileName, mask=None): 
    print("Loading nifti volume...")
    lwr_prctile = 10
    upr_prctile = 100
    denoise_wt = 150
    img = permuteOrientation(nib.load(nii_fileName))
    if mask is None:
        mask = np.zeros(img.shape)
        mask[img > 0] = 1
    img = preProcessTestVolume(img,mask, lwr_prctile, upr_prctile,denoise_wt)
    return img 

def preProcessTestVolume(img,brainMask, lwr_prctile = 10,upr_prctile =  100, denoise_wt = 80):
    ipImg = performDenoising(img,denoise_wt)
    ipImg = contrastStretching(ipImg, brainMask, lwr_prctile, upr_prctile)    
    mean_ = np.mean(ipImg[brainMask > 0])
    std_ = np.std(ipImg[brainMask>0])
    tt = ipImg - mean_
    ipImg = np.true_divide(tt,std_)
    return ipImg

def contrastStretching(img, mask, lwr_prctile, upr_prctile):
    print("Contrast stretching...")
    mm = img[mask > 0]
    p_lwr = np.percentile(mm,lwr_prctile)
    p_upr = np.percentile(mm,upr_prctile)
    opImg = rescale(img,in_range=(p_lwr,p_upr))  
    return opImg

def performDenoising(ipImg,wts = 40): 
    print("Denoising...")
    ipImg = ipImg / np.max(ipImg)  # Rescale to 0 to 1
    if wts == -1:
        print('No denoising')
    else:
        ipImg = denoise_tv(ipImg,wts)
    return ipImg

'''
permuteOrientation used here (designed for ADNI 3D FLAIR volume) load 3D volume in axial orientation. 
These scripts can be customized so that user provided 3D FLAIR volumes are loaded with axial orientation
'''
def permuteOrientation(nii):
    target_dim = (256,256)
    img_dim = nii.header.get_data_shape()
    if img_dim[1] == target_dim[0] and img_dim[2] == target_dim[1]:
        img = np.fliplr(np.rot90(nii.get_data()))
    elif img_dim[0] == target_dim[0] and img_dim[0] == target_dim[1]:
        img = np.transpose(nii.get_data(),(2,0,1))
        img = np.rot90(img,-1)
    else:
        print('Permutation not supported: ', img_dim)
    return img

def reverse_permuteOrientation(img,nii):
    target_dim = (256,256)
    img_dim = nii.header.get_data_shape()
    if img_dim[1] == target_dim[0] and img_dim[2] == target_dim[1]:
        img = np.rot90(np.fliplr(img),-1)
    elif img_dim[0] == target_dim[0] and img_dim[0] == target_dim[1]:
        img = np.rot90(img,1)
        img = np.transpose(img,(1,2,0))
        
    else:
        print('Permutation not supported: ', img_dim)
    return img

def reformat_inputOrientation(ipImg,ipType,opShape):
    '''Creates axial, sagittal, and coronal reformatting of 3D FLAIR
    and crop 3D volume to size compatible with Orthogonal Nets
    These operations can be customized based on data orientation. 
    The following script assumes ipImg is oriented axially
    '''
    if ipType is 'Axial':
        opImg = ipImg    
    elif ipType is 'Sagittal':
        opImg = np.transpose(ipImg,(2,0,1))
    elif ipType is 'Coronal':
        opImg = np.transpose(ipImg,(2,1,0))         
    else:
        print('Data orientation not supported')
    print("Creating {} test volume for Orthogonal Net".format(ipType))
    origShape = opImg.shape 
    opImg = myCrop3D(opImg,opShape)
    return opImg, origShape 

def reformat_outputOrientation(predImg,origShape,ipType):
    '''Crop 3D predictions to original image size
    and reformat orthogonal orientations to Axial.
    These operations can be customized based on data orientation
    '''
    predImg = restoreCrop3D(predImg,origShape)
    if ipType is 'Axial':
        opImg = predImg    
    elif ipType is 'Sagittal':
        opImg = np.transpose(predImg,(1,2,0))        
    elif ipType is 'Coronal':
        opImg = np.transpose(predImg,(2,1,0))        
    print("Reformatting {} prediction from Orthogonal Net: ".format(ipType))   
    return opImg

def myCrop3D(ipImg,opShape,padval=0):
    '''  Creates a 3D cropped volume from ipImg based on opShape (xDim,yDim)
    ipImg is a 3D volume    
    '''
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    if padval == 0:
        opImg = np.zeros((xDim,yDim,zDim))
    else:
        opImg = np.ones((xDim,yDim,zDim)) * np.min(ipImg)
    
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.ceil(np.abs(xPad)/2))
    x_upr = int(np.floor(np.abs(xPad)/2))
    y_lwr = int(np.ceil(np.abs(yPad)/2))
    y_upr = int(np.floor(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg


def restoreCrop3D(croppedImg,origShape):
    '''Function to restore cropped Img to origShape from which it was cropped
    '''
    
    xDim,yDim,_ = croppedImg.shape
    opImg = np.zeros(origShape)
    
    xPad = xDim - origShape[0]
    yPad = yDim - origShape[1]
    
    x_lwr = int(np.ceil(np.abs(xPad)/2))
    x_upr = int(np.floor(np.abs(xPad)/2))
    y_lwr = int(np.ceil(np.abs(yPad)/2))
    y_upr = int(np.floor(np.abs(yPad)/2))
    
    if xPad >= 0 and yPad >= 0:
        opImg = croppedImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:]
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg[x_lwr: -x_upr ,y_lwr:- y_upr,:] = croppedImg
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        
        temp_opImg= croppedImg[:,y_lwr:yDim - y_upr,:] 
        opImg[x_lwr: -x_upr,:,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = croppedImg[x_lwr:xDim - x_upr,:,:]
        opImg[:,y_lwr: -y_upr,:] = temp_opImg
    return opImg

def binarizePosterior(posterior,threshold=0.5):
# Converts posterior to a binary mask
    img = np.zeros(posterior.shape)
    img[posterior >= threshold] = 1
    img[posterior< threshold] = 0   
    return img

def createMetaCompatibleData(A,S,C):
'''
Concatenates posteriors from axial, sagittal, and coronal Orthogonal CNNs 
along the channel dimension. Assumes channel dimension is the last dim.
'''
    print("Creating input for Meta CNN...")
    A = np.expand_dims(A,axis=-1)
    S = np.expand_dims(S,axis=-1)
    C = np.expand_dims(C,axis=-1)
    Img = np.concatenate((A,S,C),axis=-1)
    Img = np.expand_dims(Img,axis=0)
    return Img

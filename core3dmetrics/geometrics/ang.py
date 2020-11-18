import os, sys
import numpy as np
import time
from osgeo import gdal
import argparse
import glob
import math
import csv
from tqdm import tqdm


def saveTiffMultiBand(outputFilename, imageData, outputDataType):
	[nBands,nRows,nCols] = imageData.shape
	
	#start creating output file
	outDriver = gdal.GetDriverByName('GTiff')
	ds = gdal.GetDriverByName('MEM').Create( '', nCols, nRows, nBands, outputDataType)#intermediary driver needed for some formats
	
	for i in range(0,nBands):#loop through all bands
		#write band data
		outband = ds.GetRasterBand(i + 1)
		outband.WriteArray(imageData[i,:,:])
	
	#create true output driver with createcopy for compatibility with more output formats
	options=[]
	outdata = outDriver.CreateCopy(outputFilename, ds, 0, options)
	
	#close down image objects
	outdata.FlushCache()#save to disk
	outdata = None
	ds.FlushCache()#save to disk
	ds = None

def saveTiffSimple(outputFilename, imageData, outputDataType):
	[nRows,nCols] = imageData.shape
	nBands=1
	
	#start creating output file
	outDriver = gdal.GetDriverByName('GTiff')
	ds = gdal.GetDriverByName('MEM').Create( '', nCols, nRows, nBands, outputDataType)#intermediary driver needed for some formats
	
	for i in range(0,nBands):#loop through all bands
		#write band data
		outband = ds.GetRasterBand(i + 1)
		outband.WriteArray(imageData)
	
	#create true output driver with createcopy for compatibility with more output formats
	options=[]
	outdata = outDriver.CreateCopy(outputFilename, ds, 0, options)
	
	#close down image objects
	outdata.FlushCache()#save to disk
	outdata = None
	ds.FlushCache()#save to disk
	ds = None

#return eigenvalues, eigenvectors, and whether all data used for statistics is valid
def getSortedEigens(points, nodataValue):
	isValid=True
	if nodataValue in points[2]: isValid=False#mark kernels containing nodata values as invalid
	if np.isnan(np.sum(points[2])): isValid=False#mark kernels containing NaN values as invalid
	
	cov=np.cov(points)#get xyz covariance
	eigenValues,eigenVectors = np.linalg.eig(cov)#do eigendecomposition
	
	#sort eigenvalues and eigenvectors by eigenvalues in descending order
	# eigenvectors are already normalized
	idx = eigenValues.argsort()[::-1]
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	
	return [eigenValues,eigenVectors,isValid]

#assumes a coordinate system with equal unit vectors is used (e.g., UTM but not Lat/Lon)
#assumes that images are already aligned and resampled to the same resolution and bounds
def computeAngleInfo(kernelRadius, pixelGSD, refDSM, refCLS, testDSM, testCLS, nodataValue, outputPath):
	start = time.time()
	
	#create circular kernel pixel mask
	irad=math.ceil(kernelRadius)
	kernel=np.zeros((irad*2+1,irad*2+1))
	for y in range(-irad,irad+1):
		for x in range(-irad,irad+1):
			if y*y+x*x <= kernelRadius*kernelRadius: kernel[y+irad,x+irad]=1
	print("Kernel mask\n", kernel)
	kernel1D=kernel.ravel()
	inds=np.where(kernel1D==1)#1D pixel indices in kernel where value==1
	xInds=np.tile(np.arange(-irad,irad+1)*pixelGSD,(irad*2+1,1))
	yInds=np.transpose(xInds)
	xPositions=xInds.ravel()[inds]
	yPositions=yInds.ravel()[inds]
	
	#storage for interesting features
	tmpData=np.zeros((9,refDSM.shape[0],refDSM.shape[1]))
	
	#find pixel coordinates where building label occurs in reference
	prows,pcols = np.where(refCLS==6)
	
	#computed angle info for pixels where it can be computed
	refSurfaceAngles=[]
	testSurfaceAngles=[]
	angleErrors=[]
	refPixelXs=[]
	refPixelYs=[]
	
	#loop through all building-labeled pixels
	print("Computing angles")
	for i in tqdm(range(0,len(prows))):#loop through all pixels with refCLS==6
	#for i in range(0,int(len(prows)*0.033)):#loop through some pixels with refCLS==6
		if prows[i]>=irad and prows[i]<refDSM.shape[0]-irad and pcols[i]>=irad and pcols[i]<refDSM.shape[1]-irad:#ignore image edges where kernel is outside bounds
			#current pixel coordinates
			x=pcols[i]
			y=prows[i]
			
			#get reference xyz values within kernel and do an eigendecomposition on the points
			nums=refDSM[y-irad:y+irad+1,x-irad:x+irad+1].ravel()[inds]#get DSM z-values within kernel around pixel
			points=np.stack((xPositions,yPositions,nums))#xyz pixel positions as [[],[],[]]
			[refEigenValues, refEigenVectors, isRefValid] = getSortedEigens(points, nodataValue)
			
			#compute reference surface statistics
			eigCurvature=refEigenValues[2]/np.sum(refEigenValues)#curvature
			eigPlanarity=(refEigenValues[1]-refEigenValues[2])/refEigenValues[0]#planarity
			
			normal=refEigenVectors[:,2]#last eigenvector is the surface normal
			if normal[2]<0: normal *= -1#flip normal if z-component is not pointing up
			
			refAngleFromUp=math.acos(np.dot(normal,[0,0,1]))*(180/math.pi)#roof slope angle in degrees
			
			if eigCurvature<0.005 and eigPlanarity>0.2:#is reference data good enough to score?
				#compute test model geometry
				nums=testDSM[y-irad:y+irad+1,x-irad:x+irad+1].ravel()[inds]#get DSM z-values within kernel around pixel
				points=np.stack((xPositions,yPositions,nums))#xyz pixel positions as [[],[],[]]
				[testEigenValues, testEigenVectors, isTestValid] = getSortedEigens(points, nodataValue)
				testNormal=testEigenVectors[:,2]#last eigenvector is the surface normal
				if testNormal[2]<0: testNormal *= -1#flip normal if z-component is not pointing up
				testAngleFromUp=math.acos(np.dot(testNormal,[0,0,1]))*(180/math.pi)#roof slope angle in degrees
				
				#angle in degrees between reference and test normals
				# note that this is the angle between normals, not the difference in slopes measured from nadir
				angleDiff=math.acos(min(np.dot(testNormal,normal),1))*(180/math.pi)
				
				#statistic values
				refSurfaceAngles.append(refAngleFromUp)
				testSurfaceAngles.append(testAngleFromUp)
				angleErrors.append(angleDiff)
				refPixelXs.append(x)
				refPixelYs.append(y)
			else:#reference data is not good enough
				isRefValid=False
				testAngleFromUp=0
				angleDiff=0
			
			#store temp data for records
			tmpData[0, y, x]=normal[0]
			tmpData[1, y, x]=normal[1]
			tmpData[2, y, x]=normal[2]
			tmpData[3, y, x]=int(isRefValid==True)#0=invalid, 1=valid
			tmpData[4, y, x]=eigCurvature
			tmpData[5, y, x]=eigPlanarity
			tmpData[6, y, x]=refAngleFromUp
			tmpData[7, y, x]=testAngleFromUp
			tmpData[8, y, x]=angleDiff
			
			# report progress every 10000 values
			#if i%10000==0:print(i*100/len(prows), "percent complete")
	
	# write angular statistics data
	myfile = open(os.path.join(outputPath,'angleData.csv'), 'w', newline='')
	wr = csv.writer(myfile)
	wr.writerow(["Pixel x", "Pixel y", "Ref angle from up","Test angle from up", "Angle error"])
	wr.writerows(np.transpose([refPixelXs, refPixelYs, np.around(np.array(refSurfaceAngles),5), np.around(np.array(testSurfaceAngles),5), np.around(np.array(angleErrors),5)]))
	
	# report scores
	angleErrors.sort()
	print("68% error value", angleErrors[int(len(angleErrors)*0.68)])
	
	# report time
	end = time.time()
	print("time =", "{:.3f}".format(end - start),"sec")
	
	# save temp data used during angle computations
	saveTiffMultiBand(os.path.join(outputPath,'angleTmpData.tif'), tmpData, gdal.GDT_Float32)
	
	# order RMS, stable slope mask, ref slope, angle error
	return [angleErrors[int(len(angleErrors)*0.68)], tmpData[3,:,:], tmpData[6,:,:], tmpData[8,:,:]]

def computeIOUs(refDSM, refDTM, refCLS, testDSM, testNDSM, testCLS, stableAngleMask, angleError, outputPath):
	start = time.time()
	
	#FP and FN label pixel counts
	FPC=0
	FNC=0
	
	#correct/incorrect masks
	correctLabel=np.zeros((refDSM.shape[0],refDSM.shape[1]))
	correctHeight=np.zeros((refDSM.shape[0],refDSM.shape[1]))
	correctAngle=np.zeros((refDSM.shape[0],refDSM.shape[1]))
	correctAGL=np.zeros((refDSM.shape[0],refDSM.shape[1]))
	
	#count TP/FP/FN components
	prows,pcols = np.where(refCLS==6)
	for y in range(0,refDSM.shape[0]):
		for x in range(0,refDSM.shape[1]):
			if refCLS[y,x]==6:
				#label
				if refCLS[y,x]==testCLS[y,x]: correctLabel[y,x]=1
				else: FNC=FNC+1
				#z error
				if abs(refDSM[y,x]-testDSM[y,x])<1: correctHeight[y,x]=1
				#angle
				if stableAngleMask[y,x]==1:
					if abs(angleError[y,x])<5: correctAngle[y,x]=1
				else:
					correctAngle[y,x]=1#assume angle is correct if no stable angle can be used to test
				#AGL
				if abs((refDSM[y,x]-refDTM[y,x])-testNDSM[y,x])<1: correctAGL[y,x]=1
			else:
				if refCLS[y,x]!=65 and testCLS[y,x]==6: FPC=FPC+1#label FP occurred

	saveTiffSimple(os.path.join(outputPath, 'delCCorrectMask.tif'), correctLabel, gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'delZCorrectMask.tif'), correctHeight, gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'delAGLCorrectMask.tif'), correctAGL, gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'delANGCorrectMask.tif'), correctAngle, gdal.GDT_Float32)
	# Combined Rasters
	saveTiffSimple(os.path.join(outputPath, 'Roof_CLS_IOU.tif'), correctLabel, gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'Roof_CLS_Z_IOU.tif'), np.multiply(correctLabel,correctHeight), gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'Roof_CLS_Z_SLOPE_IOU.tif'), np.multiply(np.multiply(correctLabel, correctHeight), correctAngle),
				   gdal.GDT_Float32)
	saveTiffSimple(os.path.join(outputPath, 'Roof_CLS_AGL_IOU.tif'),
				   np.multiply(correctLabel, correctAGL),
				   gdal.GDT_Float32)
	TPC = np.sum(correctLabel)
	TPZ = np.sum(correctHeight)
	TPAGL = np.sum(correctAGL)
	TPANG = np.sum(correctAngle)
	TPZTot = np.sum(np.multiply(correctLabel, correctHeight))
	#	TPAGLTot=np.sum(np.multiply(np.multiply(correctLabel, correctHeight), correctAGL))
	#	TPMTot=np.sum(np.multiply(np.multiply(np.multiply(correctLabel, correctHeight), correctAGL), correctAngle))
	TPAGLTot = np.sum(np.multiply(correctLabel, correctAGL))
	TPMZTot = np.sum(np.multiply(np.multiply(correctLabel, correctHeight), correctAngle))
	TPMAGLTot = np.sum(np.multiply(np.multiply(correctLabel, correctAGL), correctAngle))

	IOUC = TPC / (TPC + FPC + FNC)
	IOUZ = TPZTot / (TPC + FPC + FNC)
	IOUAGL = TPAGLTot / (TPC + FPC + FNC)
	#	IOUM=TPMTot/(TPC+FPC+FNC)
	IOUMZ = TPMZTot / (TPC + FPC + FNC)
	IOUMAGL = TPMAGLTot / (TPC + FPC + FNC)
	print("TPC", TPC)
	print("TPZ", TPZ)
	print("TPAGL", TPAGL)
	print("TPA", TPANG)
	print("TPZTot", TPZTot)
	print("TPAGLTot", TPAGLTot)
	#	print("TPMTot", TPMTot)
	print("TPMTot", TPMZTot)
	print("TPMTot", TPMAGLTot)
	print("FPC", FPC)
	print("FNC", FNC)
	print("IOUC", IOUC)  # label only
	print("IOUZ", IOUZ)  # label and z-error
	print("IOUAGL", IOUAGL)  # label and z-error and AGL-error
	#	print("IOUM", IOUM)#label and z-error and AGL-error and angle-error
	print("IOUMZ", IOUMZ)  # label and z-error and angle-error
	print("IOUMAGL", IOUMAGL)  # label and AGL-error and angle-error
	end = time.time()
	print("time =", "{:.3f}".format(end - start), "sec")
	return IOUC, IOUZ, IOUAGL, IOUMZ


def calculate_metrics(refDSM, refDTM, refCLS, testDSM, testDTM, testCLS, tform, kernel_radius=3,
					  output_path='./'):

	testNDSM = testDSM-testDTM

	# read image data
	transform = tform
	pixelWidth = transform[1]
	pixelHeight = -transform[5]

	# run angle finder and make good/bad truth mask
	[orderRMS, stableAngleMask, refSlope, angleError] = computeAngleInfo(kernel_radius, pixelWidth, refDSM, refCLS,
																		 testDSM, testCLS, -10000, output_path)

	#compute IOUs
	IOUC, IOUZ, IOUAGL, IOUMZ = computeIOUs(refDSM, refDTM, refCLS, testDSM, testNDSM, testCLS, stableAngleMask, angleError, output_path)
	return IOUC, IOUZ, IOUAGL, IOUMZ, orderRMS

if __name__ == "__main__":
	print("starting")
	
	# load ndx/dsm ref files
	dataPath = './'
	refDSM = gdal.Open(os.path.join(dataPath, 'refDSM.tif')).ReadAsArray()
	refDTM = gdal.Open(os.path.join(dataPath, 'refDTM.tif')).ReadAsArray()
	refCLS = gdal.Open(os.path.join(dataPath, 'refCLS.tif')).ReadAsArray()
	
	# load ndx/dsm test files
	testDSM = gdal.Open(os.path.join(dataPath, 'testDSM.tif')).ReadAsArray()
	testNDSM = gdal.Open(os.path.join(dataPath, 'testNDSM.tif')).ReadAsArray()
	testCLS = gdal.Open(os.path.join(dataPath, 'testCLS.tif')).ReadAsArray()
	
	# read image data
	transform = gdal.Open(os.path.join(dataPath, 'refDSM.tif')).GetGeoTransform()
	pixelWidth = transform[1]
	pixelHeight = -transform[5]
	
	# set angle-finding parameters
	kernelRadius = 3  # pixels
	# run angle finder and make good/bad truth mask
	[orderRMS, stableAngleMask, refSlope, angleError] = computeAngleInfo(kernelRadius, pixelWidth, refDSM, refCLS, testDSM, testCLS, -10000, dataPath)
	
	# compute IOUs
	computeIOUs(refDSM, refDTM, refCLS, testDSM, testNDSM, testCLS, stableAngleMask, angleError, dataPath)

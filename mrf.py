# Image denoising using MRF model
from PIL import Image
import numpy
from pylab import *

def MRF_denoise(fileName):
	im=Image.open(fileName)
	im=numpy.array(im)
	im=where (im>127,1,0) #convert to binary image
	(M,N)=im.shape	
	# Start MRF	
	(M,N)=im.shape
	y_old=im
	y=zeros((M,N))

	while(SNR(y_old,y)>0.01):
		for i in range(M):
			for j in range(N):
				index=neighbor(i,j,M,N)
				
				a=cost(1,im[i,j],y_old,index)
				b=cost(0,im[i,j],y_old,index)

				if a>b:
					y[i,j]=1
				else:
					y[i,j]=0
		y_old=y
	return y

def SNR(A,B):
	if A.shape==B.shape:
		return numpy.sum(numpy.abs(A-B))/A.size
	else:
		raise Exception("Two matrices must have the same size!")

def delta(a,b):
	if (a==b):
		return 1
	else:
		return 0

def neighbor(i,j,M,N):
	#find correct neighbors
	if (i==0 and j==0):
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	return neighbor

def cost(y,x,y_old,index):
	alpha=1
	beta=10
	return alpha*delta(y,x)+\
		beta*sum(delta(y,y_old[i]) for i in index)

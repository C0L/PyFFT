"""
  Author: Colin Drewes
  Date: 7/24/19
  Description: Testing various methods of using numpy FFT to compress image
  data. Gets compression method as first argument and image name as second
  argument. Only looks at RGB values no alpha. The compression number removes
  all smaller coeficients in the fourier transformed series.
  Usage: python FT.py [-arg] [Compression #] [ImageName]
    Args:
      -1D - Fourier transform on R/G/B values as one consecutive array of ints
      -2D - Fourier transform on R/G/B values as a 2D array
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import argparse
import math
from PIL import Image


ARG_1D = "-1D"
ARG_2D = "-2D"
ARG_2DS = "-2DS"

"""
Function Name: F_1D
Arguments: 
  data - (h, w, 3) that is going to be processed
  comp - Remove all coeficients smaller than this value
Description: This function turns the R/G/B values into one long array of
  integers and then provides a FFT on each one of those arrays. There is no
  clipping of the smaller coeficients. This is an inefficient first attempt
  to compress the image data.
"""
def F_1D(data, comp):
  
  R = []
  G = []
  B = []

  # Iterate through all rows in the image
  for row in range(h):
    # Iterate through all of the columns in the image
    for col in range(w):
      # Store all RGB values in respective array
      R.append(data[row][col][0])
      G.append(data[row][col][1])
      B.append(data[row][col][2])
  
  # Perform fourier transform as arrays as a whole
  ftR = np.fft.fft(R)
  ftG = np.fft.fft(G)
  ftB = np.fft.fft(B)
  
  for val in range(len(ftR)):
    if abs(ftR[val]) < comp:
      ftR[val] = 0
    if abs(ftG[val]) < comp:
      ftG[val] = 0
    if abs(ftB[val]) < comp:
      ftB[val] = 0

  W = np.zeros((h*w, 3), np.complex64)
  W[:, 0] = ftR
  W[:, 1] = ftG
  W[:, 2] = ftB

  # Store the data to a file to compare file sizes
  W.tofile('dataOut')
  
  # Store original image data to compare end size
  data.tofile('orig')

  # Provide the expected length to compute correct inverse FFT
  uftR = np.fft.ifft(ftR)
  uftG = np.fft.ifft(ftG)
  uftB = np.fft.ifft(ftB)
  
  # Reconstruct original image
  d = np.zeros((h, w, 3), np.uint8)
  i=0
  for row in range(h):
    for col in range(w):
      d[row][col] = [uftR[i].real, uftG[i].real, uftB[i].real]
      i+=1

  resImg = Image.fromarray(d, 'RGB')
  resImg.save('my.png')
  resImg.show()


"""
Function Name: F_2D
Arguments: 
  data - (h, w, 3) that is going to be processed
  comp - Remove all coeficients lower than this value
Description: Perform 2D FFT instead of 1D
"""
def F_2D(data, comp):
  
  R = data[:, :, 0]
  G = data[:, :, 1]
  B = data[:, :, 2]
  
  # Perform fourier transform as arrays as a whole
  ftR = np.fft.fft2(R)
  ftG = np.fft.fft2(G)
  ftB = np.fft.fft2(B)
  
  for row in range(ftR.shape[0]):
    for col in range(ftR.shape[1]):
      if abs(ftR[row, col]) < comp: 
        ftR[row, col] = 0 
 
  for row in range(ftG.shape[0]):
    for col in range(ftG.shape[1]):
      if abs(ftG[row, col]) < comp: 
        ftG[row, col] = 0 

  for row in range(ftB.shape[0]):
    for col in range(ftB.shape[1]):
      if abs(ftB[row, col]) < comp: 
        ftB[row, col] = 0 


  W = np.zeros((h, w, 3), np.complex64)
  W[:, :, 0] = ftR
  W[:, :, 1] = ftG
  W[:, :, 2] = ftB
  
  # Store the data to a file to compare file sizes
  W.tofile('dataOut1')

  # Provide the expected length to compute correct inverse FFT
  uftR = np.fft.ifft2(ftR)
  uftG = np.fft.ifft2(ftG)
  uftB = np.fft.ifft2(ftB)
  
  # Reconstruct original image
  d = np.zeros((h, w, 3), np.uint8)
  d[:, :, 0] = uftR.real
  d[:, :, 1] = uftG.real
  d[:, :, 2] = uftB.real
  
  resImg = Image.fromarray(d, 'RGB')
  resImg.save('my.png')
  resImg.show()


"""
Function Name: F_2D_S
Arguments: 
  data - (h, w, 3) that is going to be processed
  segs - Number of segements to divide image into up and down  
  comp - Remove all coeficients lower than this value
Description: Perform 2D FFT in segments as specified by user.
"""
def FT_2DS(data, comp, segs):
  R = data[:, :, 0]
  G = data[:, :, 1]
  B = data[:, :, 2]

  divY = h / segs
  divX = w / segs

#  print divY, divX

  #print (divY * segs)
  #print (divX * segs)

  ftR = np.zeros((segs, segs), np.ndarray)
  ftG = np.zeros((segs, segs), np.ndarray)
  ftB = np.zeros((segs, segs), np.ndarray)
  
  for x in range(segs):
    for y in range(segs):
      sValY = y * divY 
      eValY = ((y + 1) * divY) if (y != (segs - 1)) else (h - 1)
      
      sValX = x * divX
      eValX = ((x + 1) * divX) if (x != (segs - 1)) else (w - 1)

      ftR[y, x] = np.fft.fft2(R[sValY:eValY, sValX:eValX])
      ftG[y, x] = np.fft.fft2(G[sValY:eValY, sValX:eValX])
      ftB[y, x] = np.fft.fft2(B[sValY:eValY, sValX:eValX])

  for x in range(segs):
    for y in range(segs):
      for row in range(ftR[y][x].shape[0]):
        for col in range(ftR[y][x].shape[1]):
          if abs(ftR[y][x][row, col]) < comp: 
            ftR[y][x][row, col] = 0 
      for row in range(ftG[y][x].shape[0]):
        for col in range(ftG[y][x].shape[1]):
          if abs(ftG[y][x][row, col]) < comp: 
            ftG[y][x][row, col] = 0 
      for row in range(ftB[y][x].shape[0]):
        for col in range(ftB[y][x].shape[1]):
          if abs(ftB[y][x][row, col]) < comp: 
            ftB[y][x][row, col] = 0 
  
  #print ftR[segs, segs].shape

  ###############NEED TO FIX THIS PORTION###############################
  W = np.zeros((segs, segs, 3, ftR[segs-1, segs-1].shape[0], ftR[segs-1, segs-1].shape[1]), np.complex64)

  for x in range(segs):
    for y in range(segs):
      W[y, x, 0, :, :] = ftR[y, x]
      W[y, x, 1, :, :] = ftG[y, x]
      W[y, x, 2, :, :] = ftB[y, x]
  
  # Store the data to a file to compare file sizes
  W.tofile('dataOut2')

  uftR = np.zeros((h, w), np.uint8)
  uftG = np.zeros((h, w), np.uint8)
  uftB = np.zeros((h, w), np.uint8)
 
  for x in range(segs):
    for y in range(segs):
      sValY = y * divY 
      eValY = ((y + 1) * divY) if (y != (segs - 1)) else (h - 1)
      
      sValX = x * divX
      eValX = ((x + 1) * divX) if (x != (segs - 1)) else (w - 1)


      uftR[sValY:eValY, sValX:eValX] = np.fft.ifft2(ftR[y, x]).real
      uftG[sValY:eValY, sValX:eValX] = np.fft.ifft2(ftG[y, x]).real
      uftB[sValY:eValY, sValX:eValX] = np.fft.ifft2(ftB[y, x]).real

  # Reconstruct original image
  d = np.zeros((h, w, 3), np.uint8)
  d[:, :, 0] = uftR
  d[:, :, 1] = uftG
  d[:, :, 2] = uftB

  resImg = Image.fromarray(d, 'RGB')
  resImg.save('my.png')
  resImg.show()

compType = sys.argv[1]
compression = int(sys.argv[2])
segs = int(sys.argv[3])
imageName = sys.argv[4]


# Open provided image from file, get size and turn into array data
img = Image.open(imageName)
w, h = img.size
data = np.array(img)


if compType == ARG_1D:
  F_1D(data, compression)

if compType == ARG_2D:
  F_2D(data, compression)

if compType == ARG_2DS:
  FT_2DS(data, compression, segs)

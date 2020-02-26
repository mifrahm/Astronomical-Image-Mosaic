# Astronomical Image Mosaic
This project is the preprocessing and post processing of astronomical data that will be used in a convolutional auto encoder. More about the auto encoder can be read in this paper: [Radio Galaxy Zoo: Unsupervised Clustering of Convolutionally Auto-encoded Radio-astronomical Images](https://iopscience.iop.org/article/10.1088/1538-3873/ab213d)


## Features
This project contains the following functionalities
* Open a FITS file
* Print details of image array
* Extract square samples from image
* Reconstruct sliced samples to original image
* Reconstruct sliced samples with standard deviation
* Merge original image and standard deviation squares
* Slicing animation to illustrate the image slicing

A illustration of these functionalities can be viewed in this [Jupyter notebook](https://github.com/mifrahm/Astronomical-Image-Mosaic/blob/master/Example.ipynb).

-------

## Functionalities in Detail

### Open a FITS File
* This function is based on the [astropy](https://www.astropy.org) library which is used open a FITS file and return an image in the form of a 2D [numpy](https://numpy.org) array.

### Print Details of Image
* This basic functionality is used to print details of the image including: data, type, type, shape, dim, size, standard deviation. Along with displaying the image using the [matplotlib](https://matplotlib.org) library.

### Extract Square Samples
This function extract square samples from the original image. It reshapes the 2D image into a collection of square samples by slicing the image into the desired sample square length. This function is based on the [SKLearn](https://scikit-learn.org/). 

### Reconstruct sliced samples to original image
This function takes the sliced images and reconstructs them back to the original image.

### Reconstruct slices samples with standard deviation
This function is similar to the function above however the difference is that it calculates the standard devitation of each square and reconstructs an image based on that. This is similar to creating a heatmap to find where activity is within the image.

### Merge original image and standard devitation squares
This function takes in two image and displays merged image. This is useful to use with the original image and the reconstructed image with the standard deviation to see which area of the astronomical image has the more activity.

### Slicing animation
This function illustrates the image slicing activity and then prints out the standard deviation of each square.


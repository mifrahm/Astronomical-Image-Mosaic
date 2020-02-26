from sklearn.feature_extraction import image as image2D
from sklearn.preprocessing import MinMaxScaler
from astropy.utils.data import download_file
from itertools import product
import matplotlib.pyplot as plt
from astropy.io import fits
from copy import deepcopy
import numpy as np
import cmapy
import cv2


def calculate_stdev(samples, image_size):

    """
    
    This function generates red squares over image based on standard deviation. Based on the reconstruct_all_samples function which adds all patches and then equalises

    Parameters
    ----------
    samples : array, shape = (n_samples, sample_height, sample_width)

    image_size : tuple of ints (image_height, image_width)

    Returns
    -------
    img : array, shape = image_size
        the reconstructed image

    """

    #get length of the square sample
    p_l = samples.shape[1]
    
    #calculate ext step
    ext = p_l//2

    #create array with equal square
    image = np.zeros(image_size)
    i_h, i_w = image.shape[:2] 

    #pad image
    img = pad_image(image, p_l)

    #get the padded height and width
    pd_h, pd_w = img.shape[:2]  

    # compute the dimensions of the samples array
    n_h = pd_h - p_l + 1
    n_w = pd_w - p_l + 1
    
    #add samples to empty img array
    for p, (h, w) in zip(samples, product(range(0, n_h, ext), range(0, n_w, ext))):
        
        #add patch
        img[h:h + p_l, w:w + p_l] += np.std(p[0:p_l, 0:p_l]) #height, width

        #equalise the columns
        if (w != 0):   
            img[h:h + p_l, w:w + ext] /= 2

        #equalise the height and width overlap
        if (h != 0 and w != 0):
            img[h:h + ext, w + ext:w + p_l] /= 2
            
        #equalise the height overlap
        elif (h != 0):
            img[h:h + ext, w:w + p_l] /= 2

    #remove excess columns
    img = np.delete(img, slice(i_h, pd_h), axis=0)

    #remove excess rows
    img = np.delete(img, slice(i_w, pd_w), axis=1)

    #display image
    plt.imshow(img)
    plt.show()

    return img


def extract_samples(org_image, sample_length):
    """
    Reshape a 2D image into a collection of samples. This is based on the SKLearn library.

    The resulting samples are allocated in a dedicated array.
    - Each extracted sample is a square with equal lengths on each side
    - Extraction step is half of length.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    sample_length : the length of the desired square sample, e.g. if 100 x 100 square then enter 100.

    Returns
    -------
    samples : array, shape = (n_samples, sample_height, sample_width) or
        (n_samples, sample_height, sample_width, n_channels)
        The collection of samples extracted from the image, where `n_samples`
        is either `max_samples` or the total number of samples that can be
        extracted.

    """

    # calculate extraction step
    step = sample_length //2

    #pad image
    image = pad_image(org_image, sample_length) 

    # current image array height and width
    i_h, i_w = image.shape[:2]

    if sample_length > i_h:
        raise ValueError("Height of the sample should be less than the height"
                         " of the image.")

    if sample_length > i_w:
        raise ValueError("Width of the sample should be less than the width"
                         " of the image.")

    image = image2D.check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_samples = _extract_samples(image,
                                         sample_shape=(sample_length, sample_length, n_colors),
                                         extraction_step=step)

    samples = extracted_samples
    samples = samples.reshape(-1, sample_length, sample_length, n_colors)

    if samples.shape[-1] == 1:
        return samples.reshape((-1, sample_length , sample_length))
    else:
        return samples


def merge_images(base, overlay):
    """
    This function takes in two images and displays merged image

    Parameters
    ----------
    base: base image
    overlay: image to overlay

    """

    # scale images
    image_c = scale_image(base)
    reconstruct_c = scale_image(overlay)

    #output of combined images
    merge = cv2.addWeighted(reconstruct_c, 1, image_c, 1, 0, dtype=cv2.CV_64F)

    #display image
    plt.imshow(merge)
    plt.show()


def open_fits_file(file):
    """
    This function opens a fits file and returns and image (array)

    Parameters
    ----------
    file : fits file

    Returns
    -------
    img : 2-D numpy array

    """
    
    image_file = download_file(file, cache=True)
    img = fits.getdata(image_file)
    return img


def print_img_details(img):
    """
    This is a basic function which takes in an image and prints out all of its details and displays it.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
    (image_height, image_width, n_channels)
    The original image data. For color images, the last dimension specifies
    the channel: a RGB image would have `n_channels=3`.

    """

    print('==== Image Data ====')
    print('Image data: \n{}'.format(img))
    print('Image type: {}'.format(type(img))) 
    print('Image dtype: {}'.format(img.dtype))
    print('Image shape: {}'.format(img.shape))
    print('Image ndim: {}'.format(img.ndim))
    print('Image size: {}'.format(img.size)) 
    print('Standard Deviation {}'.format(np.std(img)))
    plt.imshow(img)
    plt.show()


def reconstruct_samples(samples, image_size):

    """

    Reconstruct the image from all of its samples.

    The reconstruct all samples functions adds all the patch values and then equalises the overlapping regions.

    Parameters
    ----------
    samples : array, shape = (n_samples, sample_height, sample_width)

    image_size : tuple of ints (image_height, image_width)

    Returns
    -------
    img : array, shape = image_size
        the reconstructed image

    """

    #get dimensions of the sample
    p_h, p_w = samples.shape[1:3]
    
    #calculate ext step
    ext = p_h//2

    #create array with equal square
    image = np.zeros(image_size)
    i_h, i_w = image.shape[:2] 

    #pad image
    img = pad_image(image, p_h)

    #get the padded height and width
    pd_h, pd_w = img.shape[:2]  

    # compute the dimensions of the samples array
    n_h = pd_h - p_h + 1
    n_w = pd_w - p_w + 1
    
    #add samples to empty img array
    for p, (h, w) in zip(samples, product(range(0, n_h, ext), range(0, n_w, ext))):
        
        #add patch
        img[h:h + p_h, w:w + p_w] += p[0:p_h, 0:p_w] #height, width

        #equalise the columns
        if (w != 0):   
            img[h:h + p_h, w:w + ext] /= 2

        #equalise the height and width overlap
        if (h != 0 and w != 0):
            img[h:h + ext, w + ext:w + p_w] /= 2
            
        #equalise the height overlap
        elif (h != 0):
            img[h:h + ext, w:w + p_w] /= 2

    #remove excess columns
    img = np.delete(img, slice(i_h, pd_h), axis=0)

    #remove excess rows
    img = np.delete(img, slice(i_w, pd_w), axis=1)

    return img



def slicing_animation(image, size):
    
    """

    Displays the animation of an image slicing along with its standard deviation for each sample

    Parameters
    ----------
    image: array, shape = (image_height, image_width)

    size : length of a single sample size. All sides of sample are assumed to be equal.


    """

    # extract samples from image
    samples = extract_samples(image, size)

    #Pad image before displaying animation
    image = pad_image(image, size) #pad image
    i_h, i_w = image.shape[:2]  # current image array height and width

    #text placement
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_corner = (10, 30)
    font_scale = 1.0
    font_color = (255, 0, 0)
    line_type = 3

    i = 0
    ext = size//2
    

    for y in range(0, i_h-ext, ext):       
        for x in range(0, i_w-ext, ext):
            intensity = int(round(np.std(samples[i])))
            alpha = intensity/10000.0

            start_point = (x, y)
            end_point = (x+size, y+size)

            # create two copies of the original image -- one for
            # the overlay and one for the final output image
            # scale image so that RGB color can be applied
            overlay = scale_image(deepcopy(image))
            output = scale_image(deepcopy(image))

            # draw a red rectangle
            cv2.rectangle(overlay, start_point, end_point,(255, 0, 0), -1)
           
            # apply the overlay
            cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

            # add text
            cv2.putText(output, "Standard Dev={}".format(np.std(samples[i])), top_corner, font, font_scale, font_color, line_type)
            
            # show the output image
            #cv2.imshow("Output", output)
            #cv2.waitKey(1)

            # show the output image
            plt.imshow(output)
            plt.show(block=False)
            plt.pause(.1)
            plt.close()
           
            i+=1



#supporting functions for animate_slice
def scale_image(img):
    
    """
    The purpose of this function is to normalize an image and apply colorMap in order to view by openCV.
    
    Parameters
    ----------
    img: array, shape = (image_height, image_width)

    Returns
    -------
    img : array, shape = image_size
    the processed image
    
    """

    #define the scale range
    scaler = MinMaxScaler(feature_range=(0, 255))

    #check to see if image can fit in ? x 3 matrix
    if (img.shape[1] % 3 != 0):
        print('not modulus 3')
        
        #calculate how many columns need to be added
        add = 3 - (img.shape[1]%3)

        #add columns
        image = np.pad(img, [(0,0),(0,add)], mode='constant', constant_values=np.average(img))

    else:
         image = img

    #reshape image array
    temp_image = image.reshape(-1, 3)
    temp_scale = scaler.fit_transform(temp_image)
    scaled = temp_scale.reshape(image.shape)

    #change to uint8 to apply ColorMap
    img = np.uint8(scaled)

    #store new image with colorMap
    #for list of colormaps, see: https://gitlab.com/cvejarano-oss/cmapy/blob/master/docs/colorize_all_examples.md
    #color = cv2.applyColorMap(img, cmapy.cmap('viridis'))
    #color = cv2.applyColorMap(img, cv2.COLORMAP_JET)  #cv2 colormap options

    return img


def pad_image(image, sample_length):

    """
    The pad_image function returns a resized version of an image to be able to contain equal amount of square samples and its extraction step.
    
    Parameters
    ----------
    image: original image. array, shape = (image_height, image_width)

    sample_length: length of square sample size

    Returns
    -------
    color : array, shape = image_size
    the processed image
    
    """

    #calculate extraction step
    extract = sample_length//2

    #by default this pads the images by zeros
    i_h, i_w = image.shape[:2] 

    n_h = extract - ((i_h - sample_length) % extract)
    n_w = extract - ((i_w - sample_length) % extract)

    #padding with the average of image
    pad_img = np.pad(image, [(0,n_h),(0,n_w)], mode='constant', constant_values=np.average(image)) 
    
    return pad_img


#supporting function for extract_samples
def calculate_samples_no(i_h, i_w, p_h, p_w):

    """Compute the number of samples that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a sample
    p_w : int
        The width of a sample
    max_samplees : integer or float, optional default is None
        The maximum number of samples to extract. If max_samples is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of samples.
    """

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_samples = n_h * n_w

    return all_samples


#supporting function for extract_samples
def _extract_samples(arr, sample_shape=8, extraction_step=1):
    """Extracts samples of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing sample position and the last n indexing
    the sample content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted samples.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which samples are to be extracted

    sample_shape : integer or tuple of length arr.ndim
        Indicates the shape of the samples to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    samples : strided ndarray
        2n-dimensional array indexing samples on first n dimensions and
        containing samples on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of samples:
        result.reshape([-1] + list(sample_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(sample_shape, image2D.numbers.Number):
        sample_shape = tuple([sample_shape] * arr_ndim)
    if isinstance(extraction_step, image2D.numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    sample_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    sample_indices_shape = ((np.array(arr.shape) - np.array(sample_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(sample_indices_shape) + list(sample_shape))
    strides = tuple(list(indexing_strides) + list(sample_strides))

    samples = image2D.as_strided(arr, shape=shape, strides=strides)

    return samples
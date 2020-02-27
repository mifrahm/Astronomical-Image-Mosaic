import image

#open fits file
img = image.open_fits_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits')

#print details of file
image.print_img_details(img)

#extract samples
samples = image.extract_samples(img, 200, 10)

#reconstruct image
reconstruct = image.reconstruct_samples(samples, img.shape, 10)

#reconstruct image with standard deviation samples
reconstruct_stdev = image.calculate_stdev(samples, img.shape, 10)

#overlay stdev samples over image
image.merge_images(img, reconstruct_stdev)

#animation with standard deviation
image.slicing_animation(img, 100, 10)


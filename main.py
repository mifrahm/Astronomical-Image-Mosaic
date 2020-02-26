import autoe

#open fits file
image = autoe.open_fits_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits')

#print details of file
autoe.print_img_details(image)

#extract samples
samples = autoe.extract_samples(image, 200, 10)

#reconstruct image
reconstruct = autoe.reconstruct_samples(samples, image.shape, 10)

#reconstruct image with standard deviation samples
reconstruct_stdev = autoe.calculate_stdev(samples, image.shape, 10)

#overlay stdev samples over image
autoe.merge_images(image, reconstruct_stdev)

#animation with standard deviation
autoe.slicing_animation(image, 100, 10)


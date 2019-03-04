from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy

########################################################################################################################
# This first section contains the input parameters for the code.

print "Initialising..."

original_image_filename = "masked_PSL_plate_1_s10268_BBXRD.tif"

result_image_file_suffix = "_result.tif"

threshold_value = 0.002

num_loops = 100  # This is the number of times the background image is created, subtracted from the original image

do_grid_cubic = True
do_grid_linear = True
do_grid_nearest = False
do_rbf_quadric = False
do_rbf_inverse = False
do_rbf_gaussian = False
do_rbf_linear = False
do_rbf_cubic = False
do_rbf_quintic = False
do_rbf_thin_plate = False

########################################################################################################################
# This section creates


def create_backgrounds(current_loop_num, input_image_filename):

    ####################################################################################################################
    # The first section reads in the pixels from the image

    raw_image_pixels = io.imread(input_image_filename)

    np.nan_to_num(raw_image_pixels)

    height, width = np.shape(raw_image_pixels)

    ####################################################################################################################
    # This section selects pixels based on the threshold value. If the pixel is below the threshold value, it is allowed
    # to stay. If the value is larger,

    gated_array = np.copy(raw_image_pixels)

    with np.nditer(gated_array, op_flags=['writeonly']) as pixels:

        for x in pixels:

            if x < threshold_value:

                x[...] = x

            else:

                x[...] = 0.0

    img = Image.fromarray(gated_array)
    img.save('loop_' + str(current_loop_num) + '_threshold.tif')
    img.close()

    raw_image_pixels = gated_array

    ####################################################################################################################
    # This section reads in the pixels from the image in such a way that can be handled by the interpolation algorithms

    points = np.empty((0, 2))
    values = np.empty((0, 1))

    shortened_points = np.empty((0, 2))
    shortened_values = np.empty((0, 1))

    print "Looping through image pixels..."

    counter = 0

    for i in range(height):

        for j in range(width):

            if raw_image_pixels[i][j] == 0.0:

                continue

            else:

                counter += 1

                points = np.append(points, [[i, j]], axis=0)
                values = np.append(values, [[raw_image_pixels[i][j]]], axis=0)

                if counter % 1000 == 0:

                    shortened_points = np.append(shortened_points, [[i, j]], axis=0)
                    shortened_values = np.append(shortened_values, [[raw_image_pixels[i][j]]], axis=0)

    ########################################################################################################################
    # Once the values are loaded, the interpolation is then undertaken.

    print "Interpolating..."

    # The following interpolate using griddata

    grid_x, grid_y = np.mgrid[0:height:1, 0:width:1]

    if do_grid_linear is True:

        grid_linear = griddata(points, values, (grid_x, grid_y), method='linear')

        grid_linear = np.reshape(grid_linear, (height, width))

        np.nan_to_num(grid_linear)

    if do_grid_nearest is True:

        grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')

        grid_nearest = np.reshape(grid_nearest, (height, width))

    if do_grid_cubic is True:

        grid_cubic = griddata(points, values, (grid_x, grid_y), method='cubic')

        grid_cubic = np.reshape(grid_cubic, (height, width))

    # The following uses the Rbf to interpolate

    if do_rbf_quadric is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_quadric = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='multiquadric')

        rbf_x_values = np.linspace(0, width, width)
        rbf_y_values = np.linspace(0, height, height)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_quadric_interp_values = func_rbf_quadric(rbf_grid_x, rbf_grid_y)

        rbf_quadric_interp_values = np.reshape(rbf_quadric_interp_values, (height, width))

    if do_rbf_inverse is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_inverse = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='inverse')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_inverse_interp_values = func_rbf_inverse(rbf_grid_x, rbf_grid_y)

        rbf_inverse_interp_values = np.reshape(rbf_inverse_interp_values, (width + 1, height + 1))

    if do_rbf_gaussian is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_gaussian = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='gaussian')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_gaussian_interp_values = func_rbf_gaussian(rbf_grid_x, rbf_grid_y)

        rbf_gaussian_interp_values = np.reshape(rbf_gaussian_interp_values, (width + 1, height + 1))

    if do_rbf_linear is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_linear = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='linear')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_linear_interp_values = func_rbf_linear(rbf_grid_x, rbf_grid_y)

        rbf_linear_interp_values = np.reshape(rbf_linear_interp_values, (width + 1, height + 1))

    if do_rbf_cubic is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_cubic = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='cubic')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_cubic_interp_values = func_rbf_cubic(rbf_grid_x, rbf_grid_y)

        rbf_cubic_interp_values = np.reshape(rbf_cubic_interp_values, (width + 1, height + 1))

    if do_rbf_quintic is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_quintic = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='quintic')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_quintic_interp_values = func_rbf_quintic(rbf_grid_x, rbf_grid_y)

        rbf_quintic_interp_values = np.reshape(rbf_quintic_interp_values, (width + 1, height + 1))

    if do_rbf_thin_plate is True:

        shortened_x_points = shortened_points[:, 0]
        shortened_y_points = shortened_points[:, 1]

        func_rbf_thin_plate = Rbf(shortened_x_points, shortened_y_points, shortened_values, function='thin_plate')

        rbf_x_values = np.linspace(0, height, height + 1)
        rbf_y_values = np.linspace(0, width, width + 1)

        rbf_grid_x, rbf_grid_y = np.meshgrid(rbf_x_values, rbf_y_values)

        rbf_thin_plate_interp_values = func_rbf_thin_plate(rbf_grid_x, rbf_grid_y)

        rbf_thin_plate_interp_values = np.reshape(rbf_thin_plate_interp_values, (width + 1, height + 1))

    ####################################################################################################################
    # Once the interpolation is complete, the result is plotted as an image.

    print "Plotting..."

    if do_grid_linear is True:

        plt.figure()
        plt.imshow(grid_linear, interpolation=None)
        plt.savefig('linear.png')
        plt.close()
        img = Image.fromarray(grid_linear)
        img.save('linear.tif')
        img.close()

    if do_grid_nearest is True:

        plt.figure()
        plt.imshow(grid_nearest, interpolation=None)
        plt.savefig('nearest.png')
        plt.close()
        img = Image.fromarray(grid_nearest)
        img.save('nearest.tif')
        img.close()

    if do_grid_cubic is True:

        plt.figure()
        plt.imshow(grid_cubic, interpolation=None)
        plt.savefig('cubic.png')
        plt.close()
        img = Image.fromarray(grid_cubic)
        img.save('cubic.tif')
        img.close()

    if do_rbf_quadric is True:

        plt.figure()
        plt.imshow(rbf_quadric_interp_values, interpolation=None)
        plt.savefig('rbf_quadric.png')
        plt.close()
        img = Image.fromarray(rbf_quadric_interp_values)
        img.save('rbf_quadric.tif')
        img.close()

    if do_rbf_inverse is True:

        plt.figure()
        plt.imshow(rbf_inverse_interp_values, interpolation=None)
        plt.savefig('rbf_inverse.png')
        plt.close()
        img = Image.fromarray(rbf_inverse_interp_values)
        img.save('rbf_inverse.tif')
        img.close()

    if do_rbf_gaussian is True:

        plt.figure()
        plt.imshow(rbf_gaussian_interp_values, interpolation=None)
        plt.savefig('rbf_gaussian.png')
        plt.close()
        img = Image.fromarray(rbf_gaussian_interp_values)
        img.save('rbf_gaussian.tif')
        img.close()

    if do_rbf_linear is True:

        plt.figure()
        plt.imshow(rbf_linear_interp_values, interpolation=None)
        plt.savefig('rbf_linear.png')
        plt.close()
        img = Image.fromarray(rbf_linear_interp_values)
        img.save('rbf_linear.tif')
        img.close()

    if do_rbf_cubic is True:

        plt.figure()
        plt.imshow(rbf_cubic_interp_values, interpolation=None)
        plt.savefig('rbf_cubic.png')
        plt.close()
        img = Image.fromarray(rbf_cubic_interp_values)
        img.save('rbf_cubic.tif')
        img.close()

    if do_rbf_quintic is True:

        plt.figure()
        plt.imshow(rbf_quintic_interp_values, interpolation=None)
        plt.savefig('rbf_quintic.png')
        plt.close()
        img = Image.fromarray(rbf_quintic_interp_values)
        img.save('rbf_quintic.tif')
        img.close()

    if do_rbf_thin_plate is True:

        plt.figure()
        plt.imshow(rbf_thin_plate_interp_values, interpolation=None)
        plt.savefig('rbf_thin_plate.png')
        plt.close()
        img = Image.fromarray(rbf_thin_plate_interp_values)
        img.save('rbf_thin_plate.tif')
        img.close()

    print "Backgrounds created!"

    return


def add_to_overall_background(total_background_image_filename, current_background_image_filename):

    total_background_image_array = io.imread(total_background_image_filename)

    current_background_image_array = io.imread(current_background_image_filename)

    np.nan_to_num(total_background_image_array)

    np.nan_to_num(current_background_image_array)

    total_background_image_array = total_background_image_array + current_background_image_array

    img = Image.fromarray(total_background_image_array)

    img.save(total_background_image_filename)

    return


def subtract_background(original_image_filename, background_image_filename, resultant_image_filename):

    original_image_array = io.imread(original_image_filename)

    background_image_array = io.imread(background_image_filename)

    np.nan_to_num(original_image_array)

    np.nan_to_num(background_image_array)

    resultant_image_array = original_image_array - background_image_array

    img = Image.fromarray(resultant_image_array)

    img.save(resultant_image_filename)

    img.close()

    return


current_starting_image_filename = original_image_filename

for i in range(num_loops):

    print "Loop " + str(i) + "..."

    result_image_filename = "loop_" + str(i) + result_image_file_suffix

    create_backgrounds(i, current_starting_image_filename)

    subtract_background(current_starting_image_filename, "linear.tif", result_image_filename)

    if i == 0:

        total_background_image_filename = "total_background.tif"

        total_background_image_starting_array = io.imread("linear.tif")

        np.nan_to_num(total_background_image_starting_array)

        img = Image.fromarray(total_background_image_starting_array)

        img.save(total_background_image_filename)

        img.close()

    else:

        add_to_overall_background(total_background_image_filename, "linear.tif")

    current_starting_image_filename = result_image_filename

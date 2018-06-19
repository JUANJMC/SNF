"""
=========================================================
Bancos de filtros de Gabor para clasificación de texturas
=========================================================

"""
from __future__ import print_function  #To bring the print function from python 3 into python 2.6+

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi #provides manipulation of n-dimencional arrays as image shift,rotate zoom it

from skimage import data #collection of algorithms for image processing
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
l13 = img_as_float(data.load('13C.png'))[shrink]
l15 = img_as_float(data.load('15C.png'))[shrink]
l16 = img_as_float(data.load('16C.png'))[shrink]
l127 = img_as_float(data.load('l127v3.png'))[shrink]
l139 = img_as_float(data.load('l139v3.png'))[shrink]
l159 = img_as_float(data.load('l159v3.png'))[shrink]

image_names = ('l13', 'l15', 'l16','l127', 'l139', 'l159')
images = (l13, l15, l16, l127, l139 , l159)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(l13, kernels)
ref_feats[1, :, :] = compute_feats(l15, kernels)
ref_feats[2, :, :] = compute_feats(l16, kernels)

print('Las imágenes giradas coinciden con las referencias que utilizan los bancos de los filtros de Gabor:')

print('original: l127, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l127, angle=30, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


print('original: l139, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l139, angle=30, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


print('original: l159, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l159, angle=0, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


print('original: l13, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l13, angle=0, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l13, rotado: 90 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l13, angle=90, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l13, rotado: 180 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l13, angle=180, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l13, rotado: 270 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l13, angle=270, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


print('original: l15, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l15, angle=0, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l15, rotado: 90 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l15, angle=90, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l15, rotado: 180 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l15, angle=180, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l15, rotado: 270 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l15, angle=270, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


print('original: l16, rotado: 0 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l16, angle=0, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l16, rotado: 90 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l16, angle=90, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l16, rotado: 180 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l16, angle=180, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: l16, rotado: 270 grados, coincidencia con: ', end='')
feats = compute_feats(ndi.rotate(l16, angle=270, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(5, 6))
plt.gray()

fig.suptitle('Respuestas de imagen para los nucleos del filtro de Gabor', fontsize=12)

axes[0][0].axis('off')


# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')



for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()

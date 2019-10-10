#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Create a datastack

# from stack import Stack
# from sentinel.sentinel_stack import SentinelStack
# from sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
# from coordinate_system import CoordinateSystem

from examples.maintest import s1_stack
from examples.maintest import track_no

import os
import math
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 10})
# import matplotlib.colors as clr
# from sklearn.preprocessing import RobustScaler

t = 0.25
t0 = 25

output_dir = '/data/Mexico/Intermediate-Output_Files/PS'


# Select and initialise Subset, slice and post-disaster image
if track_no == 22:
    slice = 'slice_500_swath_2_VV'
    postdis = '20160902'

    # Track 22 Subset - un-multilooked
    xl = 920
    xu = 1120
    yl = 21050
    yu = 21600

elif track_no == 143:
    slice1 = 'slice_500_swath_3_VV'
    slice2 = 'slice_501_swath_3_VV'
    postdis = '20170923'

    # Track 143 Subset
    xl = 550
    xu = 2500
    yl = 10400
    yu = 18500

else:
    print('Non used track number')

# Load slice info and read data for all images
# for key in s1_stack.images.keys():
#     s1_stack.images[key].load_slice_info()
#     s1_stack.images[key].slices[slice1].read_data_memmap()

# for key in s1_stack.images.keys():
#     s1_stack.images[key].load_slice_info()
#     s1_stack.images[key].slices[slice2].read_data_memmap()

# plot amplitude of an image
amplitude2 = s1_stack.images['20170923'].res_data.data_disk['amplitude']['amplitude']
plt.figure()
plt.imshow(np.log10(amplitude2), origin='lower', aspect='auto')
plt.title('Amplitude Image taken on 20170923')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.colorbar()
del amplitude2

# create an array of amplitude and intensity images over different dates
arramp = []
for key in s1_stack.images.keys():
    arramp.append(s1_stack.images[key].res_data.data_disk['amplitude']['amplitude'][xl:xu, yl:yu])
del arramp[-1]
arramp = np.array(arramp)

arrint = arramp*arramp

ampdisp_threshold = t
mean_int = np.mean(np.mean(arrint, axis=-1), axis=-1)

# To visualise stability of intensity of each image over time
# int_in_db = 10 * np.log10(mean_int / mean_int.max())
#
# plt.figure()
# plt.plot(int_in_db)
# plt.grid(True)
# plt.xlabel('Time')
# plt.ylabel('Intensity in dB')

# Do amplitude calibration because it does NOT look stable
# This method of calibration is as mentioned in the Python Module 6
nimages = s1_stack.images.__len__() - 1  # 1 less than total, excluding post disaster
calibrated_amplitude = arramp / np.sqrt(mean_int.reshape(nimages, 1, 1))
del arrint, arramp

plt.figure()
plt.imshow(np.log10(calibrated_amplitude[-1, :, :]), origin='lower', aspect='auto')
plt.title('Calibrated Amplitude Image taken on 20170923')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.colorbar()

# Calculate amplitude dispersion
calibrated_amp_dispersion = np.std(calibrated_amplitude, axis=0) / np.mean(calibrated_amplitude, axis=0)
ps_cal_pos = np.where(calibrated_amp_dispersion < ampdisp_threshold)
print('Number of calibrated PS candidates with threshold %i:' % t0, ps_cal_pos[0].size)

# Some more plots
# Plot of Persistent Scatterer Candidates After Amplitude Calibration
plt.figure()
img = plt.imshow(calibrated_amp_dispersion < ampdisp_threshold, origin='lower', aspect='auto', cmap='bone')
plt.title('Persistent Scatterer Candidates')
plt.xlabel('Range')
plt.ylabel('Azimuth')
# plt.savefig('/home/pmanivannan/Documents/plots/Amatrice_City/PS_calibrated.eps', format='eps', dpi=1000)

# Track the amplitude signal over all PSs
ps = calibrated_amp_dispersion < ampdisp_threshold
ps_ind = np.transpose(np.nonzero(ps))

# plt.figure()
# plt.imshow(ps, origin='lower', aspect='auto')
# del ps

# Latitude and Longitude
lat_sub = s1_stack.images[s1_stack.master_date].res_data.data_disk['geocode']['lat'][xl:xu, yl:yu]
lon_sub = s1_stack.images[s1_stack.master_date].res_data.data_disk['geocode']['lon'][xl:xu, yl:yu]

m = len(ps_ind) + 1
ps_positions = [[0] * 2 for i in range(m)]

ps_positions[0][0] = 'Longitude'
ps_positions[0][1] = 'Latitude'
ps_positions = np.array(ps_positions)

for i in range(len(ps_ind)):
    ps_positions[i + 1, 0] = lon_sub[ps_ind[i, 0], ps_ind[i, 1]]
    ps_positions[i + 1, 1] = lat_sub[ps_ind[i, 0], ps_ind[i, 1]]

np.savetxt(os.path.join(output_dir, 'PS_positions_%i_%i.csv' % (track_no, t0)), ps_positions, delimiter=',', fmt='%s')

# CALIBRATION OF POST DISASTER IMAGE AND STACKING DATA
amplitude = s1_stack.images[postdis].res_data.data_disk['amplitude']['amplitude'][xl:xu, yl:yu]
intensity = amplitude*amplitude

amp_meanint = np.mean(np.mean(intensity, axis=-1), axis=-1)
amp_cal = amplitude / np.sqrt(amp_meanint)
del amplitude, intensity, amp_meanint

stack_cal_amp = []
for i in range(len(calibrated_amplitude)):
    stack_cal_amp.append(calibrated_amplitude[i, :, :])
stack_cal_amp.append(amp_cal)
stack_cal_amp = np.array(stack_cal_amp)

# del amplitude_sub
nimages = len(stack_cal_amp)

# Getting the amplitude values of all PSs from all images in an array
ps_amp = np.zeros((stack_cal_amp.shape[0], ps_cal_pos[0].size, 3))

for j in range(len(stack_cal_amp)):
    for i in range(len(ps_ind)):
        ps_amp[j, i, 0] = stack_cal_amp[j, ps_ind[i, 0], ps_ind[i, 1]]
        ps_amp[j, i, 1] = ps_positions[i + 1, 0]
        ps_amp[j, i, 2] = ps_positions[i + 1, 1]


np.savetxt(os.path.join(output_dir, 'PSC_Amplitudes_%i_%i.csv' % (track_no, t0)), (ps_amp[-1, :, :]),
           delimiter=',', fmt='%s')
np.save(os.path.join(output_dir, 'PSC_Amplitudes_%i_%i' % (track_no, t0)), np.transpose(ps_amp[:, :, 0]))

# For extra validation of PSCs
# 1. Make the individual PSCs follow its mean more closely. Divide by the mean.
# 2. Check if the all the PSs have a similar trends now. Ideally, that shouldn't be the case
ps_amp_means = np.zeros((ps_amp.shape[1], 1))

for i in range(ps_amp.shape[1]):
    ps_amp_means[i, 0] = np.mean(ps_amp[:, i, 0], axis=0)

ps_amp_exp = np.zeros((ps_amp.shape[0], ps_amp.shape[1], ps_amp.shape[2]))

for i in range(ps_amp.shape[1]):
    ps_amp_exp[:, i, 0] = (ps_amp[:, i, 0]) / (ps_amp_means[i, 0])  # Division in linear but subtraction in log
    ps_amp_exp[:, i, 1] = ps_amp[:, i, 1]
    ps_amp_exp[:, i, 2] = ps_amp[:, i, 2]

np.savetxt(os.path.join(output_dir, 'PSC_Amplitudes_Norm_%i_%i.csv' % (track_no, t0)),
           np.transpose(ps_amp_exp[:, :, 0]), delimiter=',', fmt='%s')


# BOXPLOTS
# SHows the general trends of amplitude over the entire time-series
dataset = np.transpose(ps_amp[:, :, 0])
plt.figure()
plt.boxplot(dataset)
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_cal[:, :, 0])), ps_amp_cal.shape[1]))
plt.xticks(np.arange(31)+1, s1_stack.image_dates, rotation=40)
plt.xlabel('Dates')
plt.ylabel('Amplitude')
plt.title('Persistent Scatterer Amplitudes for each image in whole time series')

dataset = ps_amp[:, 1000:1500, 0]
plt.figure()
plt.boxplot(dataset)
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_cal[:, :, 0])), ps_amp_cal.shape[1]))
plt.xlabel('Pixel number')
plt.ylabel('Amplitude')
plt.title('Persistent Scatterer Amplitudes for a subset of PSs in whole time series')

# SHows the general trends of amplitude over the entire time-series after normalisation
dataset = np.transpose(ps_amp_exp[:-1, :, 0])
plt.figure()
plt.boxplot(dataset)
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_exp[:, :, 0])), ps_amp_exp.shape[1]))
plt.xticks(np.arange(24)+1, s1_stack.image_dates[:-1], rotation=40)
plt.xlabel('Dates', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.title('Normalised Persistent Scatterer Amplitudes for each image in predisaster time series')

dataset = ps_amp_exp[0:-1, 1000:1500, 0]
plt.figure()
plt.boxplot(dataset)
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_exp[:, :, 0])), ps_amp_exp.shape[1]))
plt.xlabel('Pixel number', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.title('Normalised Persistent Scatterer Amplitudes for each PS in predisaster time series')

outlier_cutoff = np.median(ps_amp_exp[0:-1, :, 0])
# outlier_cutoff = 0.6 # (track 22)

def remove_outliers(outlier_cutoff):
    ps_amp_norm = np.zeros((ps_amp_exp.shape[0], ps_amp_exp.shape[1], ps_amp_exp.shape[2]))
    for j in range(ps_amp_exp.shape[0]):
        k = 0
        for i in range(ps_amp_exp.shape[1]):
            # Subtraction in linear = division in log
            if (ps_amp_exp[:-1, i, 0].max() - ps_amp_exp[:-1, i, 0].min()) < outlier_cutoff:
                ps_amp_norm[j, k, :] = ps_amp_exp[j, i, :]
                k += 1

    ind = np.where(ps_amp_norm[0, :, 0] == 0)[0]
    ind = np.array(ind)
    ps_amp_calnorm = np.zeros((ps_amp_exp.shape[0], ind[0], 3))

    for i in range(ps_amp_exp.shape[0]):  #
        ps_amp_calnorm[i, :, :] = ps_amp_norm[i, 0:ind[0], :]
    return (ps_amp_calnorm)


ps_amp_calnorm = remove_outliers(outlier_cutoff);

# SHows the general trends of amplitude over the entire time-series after normalisation
dataset = np.transpose(ps_amp_calnorm[:-1, :, 0])
plt.figure()
plt.boxplot(dataset)
plt.xticks(np.arange(s1_stack.images.__len__()-1)+1, s1_stack.image_dates, rotation=40)
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_calnorm[:, :, 0])), ps_amp_calnorm.shape[0] + 1))
plt.xlabel('Dates', fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.title('Post Calibration: Persistent Scatterer Amplitudes for each image on whole time series \n Outlier_cutoff = '
          '%1.2f' % outlier_cutoff)

dataset = ps_amp_calnorm[:-1, 0:500, 0]
x = np.arange(1, ps_amp_calnorm.shape[1]+1, 1)
plt.figure()
plt.boxplot(dataset)
plt.plot(ps_amp_calnorm[-1, :, 0],  'ro')
# plt.plot(np.repeat(np.mean(np.mean(ps_amp_calnorm[:, :, 0])), ps_amp_calnorm.shape[1] + 1))
plt.xlabel('Pixel number',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.title('Post Calibration: Persistent Scatterer Amplitudes for each PS on the whole time series\n Outlier_cutoff ='
          ' %1.2f' % outlier_cutoff)


# THRESHOLDING - Using Gradient Method
final_array = ps_amp_calnorm
ps_amp_diff = np.zeros((final_array.shape[1], 1))

# Calculating differences of co-disaster images pairs for every PS
for i in range(final_array.shape[1]):
    # Subtraction in linear = division in log
    ps_amp_diff[i, :] = np.abs(final_array[-2, i, 0] - final_array[-1, i, 0])

psamp_diff_max = np.zeros((final_array.shape[1]))

# Maximum difference of each PS over consecutive images PRE-disaster
for i in range(final_array.shape[1]):  # this loops over all the PSs
    for j in range(final_array.shape[0] - 1):  # this loops over all images
        # if j == s1_stack.images.__len__() - 2:
        #     break
        # else:
            # diff between the same pixel in consecutive images
            diff = np.abs(
                final_array[j + 1, i, 0] - final_array[j, i, 0])  # Subtraction in linear = division in log
            if diff > psamp_diff_max[i]:
                psamp_diff_max[i] = diff

grad = np.zeros((psamp_diff_max.shape[0], 1))
for i in range(psamp_diff_max.shape[0]):
    grad[i] = ps_amp_diff[i, :] / psamp_diff_max[i]
print('Maximum Gradient:', grad.max())

# Thresholding and meta data creation
ps_damaged = np.zeros((final_array.shape[1], 3))
threshold = 1.0
for j in range(ps_amp_calnorm.shape[0]):
    for i in range(psamp_diff_max.shape[0]):  # Loop over each PS
        if ps_amp_diff[i, :] / psamp_diff_max[i] > threshold:  # Division in linear = Subtraction in log
            ps_damaged[i, 0] = (ps_amp_diff[i, :] / psamp_diff_max[i])
            ps_damaged[i, 1] = ps_amp[j, i, 1]
            ps_damaged[i, 2] = ps_amp[j, i, 2]

ps_damaged_ind = np.transpose(np.nonzero(ps_damaged[:, 0]))

# Normalise this data
for i in range(ps_damaged_ind.shape[0]):
    ps_damaged[ps_damaged_ind[i, 0], 0] = (ps_damaged[ps_damaged_ind[i, 0], 0] - threshold)/(max(ps_damaged[:, 0]) - threshold)

m = ps_damaged_ind.shape[0] + 1
ps_damaged_meta = [[0] * 3 for i in range(m)]
ps_damaged_meta[0][0] = 'Damage Level'
ps_damaged_meta[0][1] = 'Longitude'
ps_damaged_meta[0][2] = 'Latitude'
ps_damaged_meta = np.array(ps_damaged_meta)

for i in range(ps_damaged_ind.shape[0]):
    ps_damaged_meta[i + 1, :] = ps_damaged[ps_damaged_ind[i, 0], :]

np.savetxt(os.path.join(output_dir, 'Damaged_points_%i_%i.csv' % (track_no, t0)), ps_damaged_meta,\
           delimiter=',', fmt='%s')

"""
A pixel is spatially averaged with other pixels in its neighbourhood that have similar statistics.
A sliding window picks up the neighbouring pixels and the two sample Anderson-Darling test is applied to check the statistics.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



class AdaptiveAveraging:
    def __init__(self):
        pass
        # # Considering a 3*3 window to begin with - the window overlaps one of the prev. rows/columns
        # self.ml_x = 2
        # self.ml_y = 6
        # self.output_dir_files = '/home/pmanivannan/Documents/Python/Test_calibration_beta/Intermediate-Output_Files'

    def avg_withcentre(self, array1, ind, subset, ml_x, ml_y):
        a = np.zeros((array1.shape[0], len(ind)))
        ind_h0 = np.array(ind)
        for x in range(ind_h0.shape[0]):
            a[:, x] = subset[:, ind_h0[x, 0], ind_h0[x, 1]]
        centre = np.zeros((array1.shape[0], 1))
        centre[:, 0] = subset[:, ml_x//2, ml_y//2]
        a_centre = np.concatenate((a, centre), axis=1)
        return a_centre

    def normal_avg(self, array1, array2, ml_x, ml_y):
        # Average latitude/longitudes similar to adaptive averaging
        new_array1 = []
        new_array2 = []
        # Subset with centre pixel and neighbouring pixels
        subset1 = np.zeros((ml_x, ml_y))
        subset2 = np.zeros((ml_x, ml_y))
        cx = 0
        cy = 0
        ml_yy = int(ml_y - (ml_y/2))
        for i in range(0, array1.shape[0], ml_x):
            if (i + ml_x) > array1.shape[0]:
                break
            else:
                cx += 1
                for j in range(0, array1.shape[1], ml_yy):
                    if (j + ml_y) > array1.shape[1]:
                        break
                    else:
                        cy += 1
                        # Create a (no_of_images x 3 x 3) subset
                        subset1 = array1[i:i + ml_x, j:j + ml_y]
                        subset2 = array2[i:i + ml_x, j:j + ml_y]
                        new_array1.append(np.mean(subset1.flatten()))
                        new_array2.append(np.mean(subset2.flatten()))
            print('Line', i + 1)

        new_array1 = np.array(new_array1)
        new_array2 = np.array(new_array2)

        final_array1 = np.reshape(new_array1, (cx, cy // cx))
        final_array2 = np.reshape(new_array2, (cx, cy // cx))

        return final_array1, final_array2


    def adaptive_avg(self, array1, array2, ml_x, ml_y):

        # Adaptively multi-looked amplitude
        new_array1 = []
        new_array2 = []

        # Subset with centre pixel and neighbouring pixels
        # subset1 = np.zeros((array1.shape[0], ml_x, ml_y))
        # subset2 = np.zeros((array1.shape[0], ml_x, ml_y))
        cx = 0
        cy = 0
        for i in range(0, array1.shape[1], ml_x):
            cx += 1
            for j in range(0, array1.shape[2], ml_y):
                cy += 1
                # Create a (no_of_images x ml_x x ml_y) subset
                subset1 = array1[:, i:i+ml_x, j:j+ml_y]
                subset2 = array2[:, i:i+ml_x, j:j+ml_y]
                ind_h0 = []
                ind_h1 = []
                for ii in range(ml_x):
                    for jj in range(ml_y):
                        # Don't check centre pixel with centre pixel
                        if not (ii == ml_x//2 and jj == ml_y//2):
                            # Statistical similarity test
                            resultAnderson = st.anderson_ksamp([subset2[:-1, ii, jj], subset2[:-1, ml_x//2, ml_y//2]])
                            # pixels are from the same distribution with a significance level of 1%
                            if resultAnderson.significance_level < 0.01:
                                # print("Pixel [%i, %i]  is similar to centre pixel" % (ii, jj))
                                ind_h0.append([ii, jj])
                            else:
                                # print("Pixel [%i, %i]  is not similar to centre pixel" % (ii, jj))
                                ind_h1.append([ii, jj])
                if len(ind_h0) > len(ind_h1):
                    new_array1.append(np.mean(self.avg_withcentre(array1, ind_h0, subset1, ml_x, ml_y), axis=1))
                    new_array2.append(np.mean(self.avg_withcentre(array2, ind_h0, subset2, ml_x, ml_y), axis=1))
                else:
                    new_array1.append(np.mean(subset1, axis=(1, 2)))
                    new_array2.append(np.mean(subset2, axis=(1, 2)))
            print('Line', i+1)
        new_array1 = np.transpose(np.array(new_array1))
        new_array2 = np.transpose(np.array(new_array2))
        final_array1 = np.reshape(new_array1, (array1.shape[0], cx, cy//cx))
        final_array2 = np.reshape(new_array2, (array1.shape[0], cx, cy//cx))

        return final_array1, final_array2


if __name__ == "__main__":
    ml_x = 2
    ml_y = 6
    aa = AdaptiveAveraging()
    # adpt_amp, adpt_int = aa.adaptive_avg(arramp, arrint, ml_x, ml_y)
    # np.save(os.path.join(output_dir_files + 'AdptAvg_2x6'), final_arramp)
    # final_array = np.load('/home/pmanivannan/Documents/Python/Test_calibration_beta/Intermediate-Output_Files/AdptAvg_2x6.npy')

    plt.close('all')

    # fig = plt.figure()
    # for i in range(4):
    #     ax = fig.add_subplot(2, 2, i + 1)
    #     # plt.subplot(5, 2, i+1)
    #     im = plt.imshow(np.log10(arramp[0, i, :, :]), origin='lower', aspect='auto')
    #     ax.title.set_text(s1_stack.image_dates[i])
    #     ax.set_xlabel('Range')
    #     ax.set_ylabel('Azimuth')
    #     plt.colorbar()
    #     plt.show()
    # fig.suptitle('Non-Spatially-averaged amplitudes')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # fig = plt.figure()
    # ax2 = fig.add_subplot(1, 1, 1)
    # plt.imshow(np.log10(final_arramp[0, -1, :, :]), origin='lower', aspect='auto')
    # plt.colorbar()
    # plt.title("Single image - not averaged")


    # fig = plt.figure()
    # for i in range(4):
    #     ax1 = fig.add_subplot(2, 2, i+1)
    #     im = plt.imshow(np.log10(final_arramp[i, :, :]), origin='lower', aspect='auto')
    #     ax1.title.set_text(s1_stack.image_dates[i])
    #     ax1.set_xlabel('Range')
    #     ax1.set_ylabel('Azimuth')
    #     plt.colorbar()
    # fig.suptitle('Spatially averaged amplitudes')

    plt.figure()
    plt.imshow(np.log10(adpt_amp[-1, :, :]), origin='lower', aspect='auto')
    plt.colorbar()
    plt.title('Adaptively averaged amplitude 2x6 taken on 20160902')

    plt.figure()
    plt.imshow(np.log10(adpt_int[-1, :, :]), origin='lower', aspect='auto')
    plt.colorbar()
    plt.title('Adaptively averaged amplitude 2x6 taken on 20160902')
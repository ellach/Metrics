import torch
import numpy as np
import csv
import os
import cv2
import skimage.measure
import itertools
import matplotlib.pyplot as plt
from brisque import BRISQUE
from TM import TMQI
import seaborn as sns
from scipy.stats import wilcoxon
import skimage.measure
import ast
from skimage.metrics import structural_similarity
from math import log10, sqrt
from skvideo.measure import niqe


def TENENGRAD(dataset, path_to_results, ksize=3, file_name=''):
    """
    Computes the Tenengrad gradient magnitude of an image.

    :param dataset: A grayscale images.
    :param ksize: The kernel size.
    :param path_to_results: The path to save CSV results.
    :param file_name: The file name of saved CSV results.
    :return: The Tenengrad gradient magnitude.
    """

    results = []
    for img in dataset:

        Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
        Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
        FM = Gx * Gx + Gy * Gy
        mn = cv2.mean(FM)[0]
        if np.isnan(mn):
            results.append(FM)
            return np.nanmean(FM)
        else:
            results.append(mn)
    write_csv(path_to_results, results, file_name)
    return results


def TMQI_(hdr, ldr, path_to_results, file_name=''):
    """
    TMQI: Tone Mapped Image Quality Index.

    :param hdr: the HDR image being used as reference.
    :param ldr: the LDR image being compared.
    :param path_to_results: The path to save CSV results.
    :param file_name: The file name of saved CSV results.
    :return: The scores of the TMQI metric for each image.
    """
    results = []

    for (h, l) in zip(hdr, ldr):
        Q, S, N, s_local, s_maps = TMQI()(h, l)

        results.append(Q)
    write_csv(path_to_results, results, file_name)
    return results


def BRISQUE_(dataset, path_to_results, file_name=''):
    """
     BRISQUE metric.

     :param dataset: The grayscale image dataset.
     :param path_to_results: The path to save CSV results.
     :param file_name: The file name of saved CSV results.
     :return: The scores of the BRISQUE metric for each image.
     """
    obj = BRISQUE(url=False)
    scores = []
    for i, data in enumerate(dataset):
        scores.append(obj.score(data))
    write_csv(path_to_results, scores, file_name)
    return scores


def calc_wilcoxon(d1, d2, a):
    """
     Wilcoxon rank test.

     :param d1: The first set of measurements.
     :param d2: The second set of measurements.
     :param a: The alternative hypothesis.
     :return:  The p-value (p) and the sum of the ranks of the differences above zero (w).
     """
    w, p = wilcoxon(d1, d2, alternative=a)
    return w, p


def PSNR(ref, compressed):
    """
     PSNR metric.

     :param ref: The reference images.
     :param compressed: The enhanced images.
     :return: The scores of the PSNR metric for each image.
     """
    mse = np.mean((ref - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM_(img1, img2):
    """
     SSIM metric.

     :param img1: The reference image.
     :param img2: The enhanced image.
     :return: The scores of the SSIM metric for each image.
     """
    score, diff = structural_similarity(img1, img2, full=True)
    return score


def NIQE_(dataset):
    """
     Function calculates Naturalness Image Quality Evaluator (NIQE).

     :param dataset: The grayscale batch of  images.
     :return: The scores of the NIQE metric for each image.
     """
    scores = []
    for data in dataset:
        data = np.expand_dims(data, axis=[0, -1])
        score = niqe(data)
        scores.append(score)
    return scores


def get_entropy_csv(dataset, path_to_results, file_name=''):
    """
     Entropy metric.

     :param dataset: The grayscale batch of images.
     :param path_to_results: The path to save CSV results.
     :param file_name: The file name of saved CSV results.
     :return: The scores of the entropy metric for each image.
     """
    scores = []
    for data in dataset:
        scores.append(skimage.measure.shannon_entropy(data))
    write_csv(path_to_results, scores, file_name)
    return scores


def write_csv(path, data, file_name=''):
    """
     Function create and write in to CSV file.

     :param path: The path to dataset.
     :param data:
     :param file_name: The path to save CSV results.
     :return:
     """
    f = open(path + file_name + '.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def read_csv(path):
    """
     Function read CSV file .

     :param path: The path to the dataset of images.
     :return: The dataset as 2D array.
     """
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    output = np.array(["{:.3f}".format(ast.literal_eval(item)) for item in list(itertools.chain.from_iterable(data))])
    return output


def get_avg_and_std_batch(imgs):
    """
     Function return mean adn STD of the batch of data.

     :param imgs: The batch of images.
     :return: The mean and STD of the batch .
     """
    return torch.mean(imgs), torch.std(imgs)


def open_dataset(path):
    """
     Open dataset.

     :param path: The path to the dataset.
     :return: The set of image 2D arrays.
     """
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')])

    dataset = [cv2.normalize(cv2.cvtColor(j, cv2.COLOR_BGR2GRAY), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_32F) for j in [cv2.imread(i) for i in image_files]]

    return dataset


def CLAHE(files, path_to_results, path_to_data=''):
    """
     CLAHE method.

     :param files: the HDR image being used as reference.
     :param path_to_results: the LDR image being compared.
     :param path_to_data: The path to save CSV results.
     :return:
     """
    for f in files:
        image = cv2.imread(f)

        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0)
        final_img = clahe.apply(image_bw)

        img_name = f.split(path_to_data)[-1]
        plt.imsave(path_to_results + img_name, final_img, cmap=plt.cm.gray)


def get_boxplot(metrics_datasets, xticklabels, ylabel, ncols):
    """
     Function creates boxplot chart.

     :param metrics_datasets: The arrays with metric scors for each chart.
     :param xticklabels: The array of labels for x-axes in each chart.
     :param ylabel: The array of labels for y-axes in each chart.
     :param ncols: The number of columns in chart.
     :return:
     """
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(18, 8.5))
    fig.tight_layout(pad=7)

    axes = axes.ravel()

    for i in range(ncols):
        axes[i].yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)
        sns.boxplot(data=metrics_datasets[i], ax=axes[0], showfliers=False, width=0.6)
        axes[i].set_ylabel(ylabel[i], fontsize=18, weight='bold')
        axes[i].set_xticklabels(xticklabels, rotation=30, fontsize=18, weight='bold')

        # Create a strip plot on top of the box plot
        sns.stripplot(data=metrics_datasets[i], ax=axes[0], color="black", size=1)
    plt.savefig('boxplot.png')


def get_wilcoxon(metrics_datasets, xticklabels, ylabel, ncols):
    """
      Function creates bar and whisker plot.

     :param metrics_datasets: The arrays with metric scors for each plot.
     :param xticklabels: The array of labels for x-axes in each plot.
     :param ylabel: The array of labels for y-axes in each plot.
     :param ncols: The number of columns in plot.
     :return:
     """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(11, 5))
    fig.tight_layout(pad=7.0)
    axes = axes.ravel()  # array to 1D

    plt.figtext(0.01, -0.1, "*Significant at p-val < 0.05", ha="center", fontsize=14)

    colors = ['blue', 'lightseagreen', 'seagreen', 'grey', 'lightpink']

    for i in range(ncols):

        CTEs_ = [np.mean(metrics_datasets[i][0]), np.mean(metrics_datasets[i][1]), np.mean(metrics_datasets[i][2]),
                 np.mean(metrics_datasets[i][3]), np.mean(metrics_datasets[i][4])]

        error_ = [np.std(metrics_datasets[i][0]), np.std(metrics_datasets[i][1]), np.std(metrics_datasets[i][2]),
                  np.std(metrics_datasets[i][3]), np.std(metrics_datasets[i][4])]

        axes[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.2)
        axes[i].bar([1, 2, 3, 4, 5], CTEs_,
                    yerr=error_,
                    align='center',
                    alpha=0.5,
                    color=colors,
                    ecolor='black',
                    capsize=10)

        axes[i].set_ylabel(ylabel[i], fontsize=18)
        axes[i].set_xticks([1, 2, 3, 4, 5])
        axes[i].set_xticklabels(xticklabels[i], rotation=65, fontsize=18)

        c21 = max(max(metrics_datasets[i][0]), max(metrics_datasets[i][4]))
        c22 = max(max(metrics_datasets[i][1]), max(metrics_datasets[i][4]))
        c23 = max(max(metrics_datasets[i][2]), max(metrics_datasets[i][4]))
        c24 = max(max(metrics_datasets[i][3]), max(metrics_datasets[i][4]))

        axes[i].plot([1, 1, 5, 5], [c21 - 0.02, c21, c21, c21 - 0.02], lw=1, c='k')
        axes[i].plot([2, 2, 5, 5], [c22 - 0.02, c22, c22, c22 - 0.02], lw=1, c='k')
        axes[i].plot([3, 3, 5, 5], [c23 - 0.02, c23, c23, c23 - 0.02], lw=1, c='k')
        axes[i].plot([4, 4, 5, 5], [c24 - 0.02, c24, c24, c24 - 0.02], lw=1, c='k')

        axes[i].text((1 + 5) * .5, c21 + 0.001, '*', ha='center', va='bottom', fontsize=10)
        axes[i].text((2 + 5) * .5, c22 + 0.001, '*', ha='center', va='bottom', fontsize=10)
        axes[i].text((3 + 5) * .5, c23 + 0.001, '*', ha='center', va='bottom', fontsize=10)
        axes[i].text((4 + 5) * .5, c24 + 0.001, '*', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    plt.savefig('wilcoxon.png')





import nibabel as nb 
import numpy as np

import os

from nibabel.filebasedimages import ImageFileError
from scipy.io import loadmat 
from scipy.stats import zscore
from scipy.signal import butter, sosfiltfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def convert_2d_nifti(mask_bin, nifti_data):
    nonzero_indx = np.nonzero(mask_bin)
    nifti_2d = nifti_data[nonzero_indx]
    return nifti_2d.T


def initialize_matrix(fps, nscans, file_format, mask, verbose):
    if file_format == 'nifti':
        n_ts = len(np.nonzero(mask)[0])
    elif file_format == 'cifti':
        cifti = nb.load(fps[0])
        n_ts = cifti.shape[1]
    elif file_format == 'txt':
        txt = np.loadtxt(fps[0])
        n_ts = txt.shape[1]
    elif file_format == '1D':
        txt = np.loadtxt(fps[0])  
        n_ts = txt.shape[1]
    n_t = 0
    for fp in fps:
        if file_format == 'nifti':
            nifti = nb.load(fp)
            n_t += nifti.header['dim'][4]
        elif file_format == 'cifti':
            cifti = nb.load(fp)
            n_t += cifti.shape[0]
        elif file_format == 'txt':
            n_t += sum(1 for _ in open(fp))
    if verbose:
        print(f'initializing matrix of size ({n_t}, {n_ts})')
    matrix_init = np.zeros((n_t, n_ts))
    return matrix_init

def load_and_clean_1D(file_path):
    clean_data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace(',', ' ')
            line = line.strip()
            parts = line.split()
            
            try:
                float_line = [float(x) for x in parts]
                clean_data.append(float_line)
            except ValueError:
                continue
    return np.array(clean_data)

def load_data(input_files, file_format, mask_fp, normalize, 
              bandpass, low_cut, high_cut, tr, verbose):
    with open(input_files, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
        fps = [line for line in lines if line]
    if not fps:
        raise ValueError("No valid file paths found in the input file.")
    parameter_check(fps, file_format, tr, bandpass, mask_fp)
    mask = None
    header = None
    if file_format.lower() == '1d':
        data = [load_and_clean_1D(fp) for fp in fps] 
    else:
        data, mask, header = load_scans(
            fps, file_format, mask_fp, normalize, bandpass, 
            low_cut, high_cut, tr, verbose
        )

    return data, mask, header if file_format.lower() != '1d' else (data, None, None)

def load_scans(fps, file_format, mask_fp, normalize, 
               bandpass, low_cut, high_cut, tr, verbose):
    n_scans = len(fps)
    if file_format == 'nifti':
        mask = nb.load(mask_fp)
        mask_bin = mask.get_fdata() > 0
    else:
        mask = None
        mask_bin = None

    group_data = initialize_matrix(fps, n_scans, file_format, 
                                   mask_bin, verbose) 
    print(f'loading and concatenating {n_scans} scans')
    if bandpass and verbose:
        print(
          f'bandpass filtering of signals between {low_cut} - {high_cut} Hz '
          ' will be performed'
        )
    indx=0
    for fp in fps:
        data, header = load_file(fp, file_format, mask_bin, 
                                 bandpass, low_cut, high_cut,
                                 tr, verbose)
        data_n = data.shape[0]
        if normalize == 'zscore':
            data = zscore(data, nan_policy='omit')
        elif normalize == 'mean_center':
            data = data - np.mean(data, axis=0)
        data = np.nan_to_num(data)
        group_data[indx:(indx+data_n), :] = data
        indx += data_n
    return group_data, mask, header

def load_file(fp, file_format, mask, bandpass, 
              low_cut, high_cut, tr, verbose):
    if file_format == 'nifti':
        nifti = nb.load(fp)
        nifti_data = nifti.get_fdata()
        data = convert_2d_nifti(mask, nifti_data)
        header = nifti.header
    elif file_format == 'cifti':
        cifti = nb.load(fp)
        data = cifti.get_fdata()
        header = cifti.header
    elif file_format == 'txt':
        data = np.loadtxt(fp)
        header = None 
    elif file_format == '1D':
        func_data = []
        with open(fp, 'r') as f:
            try:
                for line in f:
                    func_data.append([float(x) for x in line.split()])
            except ValueError as e:
                raise Exception(f"Error reading file {fp}: {str(e)}")
        func_data = np.array(func_data)
        mask = None
        header = None

    if bandpass:
        npad=1000
        fs = 1/tr 
        sos = butter_bandpass(low_cut, high_cut, fs)
        data_pad = np.pad(data,[(npad, npad), (0, 0)], 'median')
        data_filt = sosfiltfilt(sos, data_pad, axis=0)
        data = data_filt[npad:-npad, :]
    return data, header

def parameter_check(fps, file_format, tr, bandpass, mask):
    if (mask is None) and (file_format == 'nifti'):
        raise Exception('A mask .nii file must be supplied when the file format is nifti')
    if bandpass and (tr is None):
        raise Exception('The TR must be supplied if bandpass=True')
    try:
        if file_format == '1D':
            with open(fps[0], 'r') as f:
                for line in f:
                    line = line.replace(',', ' ').strip()
                    if line.startswith('#') or not line:
                        continue
                    try:
                        _ = [float(x) for x in line.split()]
                        break  
                    except ValueError:
                        continue  
            actual_format = '1D'
        else:
            data_obj = nb.load(fps[0])
            if isinstance(data_obj, nb.nifti1.Nifti1Image):
                actual_format = 'nifti'
            elif isinstance(data_obj, nb.cifti2.cifti2.Cifti2Image):
                actual_format = 'cifti'
            else:
                raise Exception("Unknown image format")
    except (ImageFileError, Exception) as e:
        raise Exception(f"Error loading file: {str(e)}")
    if file_format != actual_format:
        raise Exception(f"It looks like the file format specified: '{file_format}' "
                        f"does not match the file format of the input files: '{actual_format}'")

def read_input_file(input_files):
    with open(input_files, 'r') as file:
        fps = [line.rstrip() for line in file]
        fps = [line for line in fps if len(line)>0]
    return fps


def write_out(data, mask, header, file_format, out_prefix):
    if file_format == 'nifti':
        mask_bin = mask.get_fdata() > 0
        nifti_4d = np.zeros(mask.shape + (data.shape[0],), 
                            dtype=data.dtype)
        nifti_4d[mask_bin, :] = data.T
        nifti_out = nb.Nifti2Image(nifti_4d, mask.affine)
        nb.save(nifti_out, f'{out_prefix}.nii')
    elif file_format == 'cifti':
        # https://neurostars.org/t/alter-size-of-matrix-for-new-cifti-header-nibabel/20903/2
        # first get tr from Series axis
        tr = header.get_axis(0).step
        ax_0 = nb.cifti2.SeriesAxis(0, tr, data.shape[0]) 
        ax_1 = header.get_axis(1)
        new_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
        cifti_out = nb.cifti2.Cifti2Image(data, new_header)
        nb.save(cifti_out, f'{out_prefix}.dtseries.nii')



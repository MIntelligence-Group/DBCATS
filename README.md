# DB-CATS: Dynamic Brain Connectivity Across Time Scales Using Complex Principal Component Analysis

## Introduction
DB-CATS is a sophisticated tool designed for the analysis of fMRI scan data, employing Complex Principal Component Analysis (CPCA) to unravel dynamic brain connectivity across various time scales. This repository provides scripts and methods crucial for processing and analyzing fMRI data, facilitating neuroscience and cognitive science research.

## Methods
The core of DB-CATS lies in its application of CPCA, a method that extends traditional PCA into the complex domain, allowing for the analysis of both magnitude and phase information in fMRI data. This is particularly essential for capturing the dynamic nature of brain connectivity, which standard PCA might overlook.

### Key Features:
- **Bandpass Filtering**: For noise reduction and signal enhancement.
- **Normalization**: Options include z-score normalization to standardize data.
- **Dynamic Phase Maps**: Construction of time-varying phase maps to illustrate connectivity changes.

## Installation and Usage

### Cloning the Repository
To get started with DB-CATS, clone the repository to your local machine:
```sh
git clone https://github.com/MIntelligence-Group/DBCATS.git
cd DBCATS
```

### Installing Dependencies
Install the required libraries specified in the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

### Running the Script
The main script for the analysis is `cpca.py`, located in the root directory. Additional scripts `load_write.py` and `cpca_reconstruction.py` are in the `utils` directory.

#### Command Breakdown
The following command demonstrates how to execute the CPCA analysis:

```sh
python cpca.py -i input_files.txt -n 7 -f 1D -t complex -o output_filename -r varimax -recon --normalize zscore -b --bandpass_filter_low 0.01 --bandpass_filter_high 0.1 --sampling_unit 2 --n_recon_bins 1200
```

##### Command Options:
- `-i input_files.txt`: Specifies the input file containing paths to fMRI data files.
- `-n 7`: Number of principal components to retain.
- `-f 1D`: File format of the input data (e.g., 1D for Matlab 1D files).
- `-t complex`: Type of PCA to perform (complex in this case).
- `-o output_filename`: Prefix for the output files.
- `-r varimax`: Rotation method to apply to the principal components.
- `-recon`: Flag to perform signal reconstruction.
- `--normalize zscore`: Method to normalize the data (z-score normalization).
- `-b`: Flag to apply bandpass filtering.
- `--bandpass_filter_low 0.01`: Low cut-off frequency for bandpass filter.
- `--bandpass_filter_high 0.1`: High cut-off frequency for bandpass filter.
- `--sampling_unit 2`: Sampling unit (TR) in seconds.
- `--n_recon_bins 1200`: Number of bins for phase reconstruction.

### Example for .1D Files
To run the analysis, ensure your `input_files.txt` contains the paths to your files in the specified format(.1D in our case) and execute the above command in the terminal.

## Conclusion
DB-CATS is a powerful tool for researchers looking to delve into the complexities of brain connectivity through fMRI data. By utilizing CPCA, this toolkit provides a nuanced view of dynamic connectivity patterns, paving the way for advanced neurological insights.

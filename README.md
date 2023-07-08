# A Deep Learning-based method for Multi-Zone Sound Field Synthesis

Code repository for my master thesis on _A Deep Learning-based method for Multi-Zone Sound Field Synthesis_.

- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Dependencies
- Python, has been tested with version 3.6.9
- Numpy, sci-kit-image, sci-kit-learn, tqdm, matplotlib
- Jax
- Tensorflow 2.4.1
- [sfs](https://sfs-python.readthedocs.io/en/0.6.2/)

### Data generation
There are two different scripts to generate the data, _generate_test_data_linear_array.py_ and  _generate_train_data_linear_array.py_. The parameters used for data generation (e.g. sources position, array position) are defined into _data_lib/params_linear_.

The command-line arguments are the following
- gt_soundfield: bool, True if want to generate also data related to ground truth sound field
- dataset_path: String, folder where to store the dataset

### Network training
There is a script to generate the data with the linear array, namely _generate_data_linear_array.py_. The parameters used for data generation (e.g. sources position, array position) are defined into _data_lib/params_linear_.
- epochs: Int, number of epochs 
- batch_size: Int, dimension of batches used for training
- log_dir: String, folder where store training logs accessible via [Tensorboard](https://www.tensorflow.org/tensorboard)
- gt_soundfield_dataset_path: String, path to numpy array where ground truth sound-field data are contained
- learning_rate: Float, learning rate used for the optimizer
- green_function: String, path to numpy array where green function between secondary sources and evaluation points is contained

### Results computation
To compute Reproduction Error (MSE) and Structural Similarity Index (SSIM), run the code into _plot_results.py_ with the following arguments

- dataset_path: String, path to the folder to save results stored into arrays.
- array_type: String, type of array, i.e. linear or circular

N.B. pre-trained models used to compute the results can be found in folder _models_s

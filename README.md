![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-v.%203.7.4-blue)
![pytorch](https://img.shields.io/badge/pytorch-v.%201.0.0-orange)
![license](https://img.shields.io/badge/license-CC_BY_4.0-yellow)



# De novo molecule design with chemical language models

In this repository, you will find a hands-on tutorial to generate focused libraries using RNN-based chemical language models.<div>
The code for the following two methods is provided:
* **Bidirectional Molecule Design by Alternate Learning** (BIMODAL), designed for SMILES generation – see [Grisoni *et al.* 2020](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943).
* **Forward RNN**, *i.e.*, "classical" unidirectional RNN for SMILES generation. 
In addition to the method code, several pre-trained models are included.

**Note!** 
This repository contains the code for the hands-on chapter: "De novo Molecule Design with Chemical Language Models" and has a teaching purpose only. <div>
To use the most up-to-date versions of the methods, have a look at the following repositories:
* https://github.com/ETHmodlab/BIMODAL for BIMODAL.
* https://github.com/ETHmodlab/virtual_libraries for unidirectional RNNs.

Happy coding!


## Table of Contents
1. [Getting started](#Prerequisites)
2. [Using the Code](#Using_the_code)
    1. [Provided Jupyter notebook](#notebook)
    2. [Sampling from a pre-trained model](#Sample)
    3. [Fine-tuning a model on your data](#Finetuning) 
    4. [Data pre-processing](#preprocessing) 
3. [Advanced functions](#advanced)
4. [Authors](#Authors)
5. [License](#License)
6. [How to cite](#cite) 

## Getting started <a name="Prerequisites"></a>

This repository can be cloned with the following command:

```
git clone https://github.com/ETHmodlab/de_novo_design_RNN
```

To install the necessary packages to run the code, we recommend using [conda](https://www.anaconda.com/download/). 
Once conda is installed, you can create the virtual environment as follows:

```
cd path/to/repository/
conda env create -f environment.yml
```

To activate the dedicated environment:
```
conda activate de_novo
```

Your code is now ready to use!

# Using the code <a name="Using_the_code"></a>


## Provided Jupyter notebook <a name="notebook"></a>

In this repository, you can find a Jupyter notebook that will help you get started with using the code. We recommend having a 
look at the notebook first. <div>

To use the provided notebook, move to the “example” folder and launch the Jupyter Notebook application, as follows:
```
cd example
jupyter notebook
```

A webpage will open, showing the content of the “code” folder. 
Double clicking on the file [“de_novo_design_pipeline.ipynb”](example/de_novo_design_pipeline.ipynb) opens the notebook. <div>
Each line of the provided code can be executed to visualize and reproduce the results of this tutorial. 
Below, you will also find some additional details into more advanced setting tuning.

## Sampling from a pre-trained model <a name="Sample"></a>

In this repository, we provide you with 22 pre-trained models you can use for sampling (stored in [evaluation/](evaluation/)).
These models were trained on a set of 271,914 bioactive molecules from ChEMBL22 (K<sub>d/I</sub>/IC<sub>50</sub>/EC<sub>50</sub> <1μM), for 10 epochs.    

To sample SMILES, you can create a new file in [model/](model/) and use the *Sampler class*. 
For example, to sample from the pre-trained BIMODAL model with 512 units:

```
from sample import Sampler
experiment_name = 'BIMODAL_fixed_512'
s = Sampler(experiment_name)
s.sample(N=100, stor_dir='../evaluation', T=0.7, fold=[1], epoch=[9], valid=True, novel=True, unique=True, write_csv=True)
```

Parameters:
* *experiment_name* (str): name of the experiment with pre-trained model you want to sample from (you can find pre-trained models in [evaluation/](evaluation/))
* *stor_dir* (str): directory where the models are stored. The sampled SMILES will also be saved there (if write_csv=True)
* *N* (int): number of SMILES to sample
* *T* (float): sampling temperature
* *fold* (list of int): number of folds to use for sampling
* *epoch* (list of int): epoch(s) to use for sampling
* *valid* (bool): if set to *True*, only generate valid SMILES are accepted (increases the sampling time)
* *novel* (bool): if set to *True*, only generate novel SMILES (increases the sampling time)
* *unique* (bool): if set to *True*, only generate unique SMILES are provided (increases the sampling time)
* *write_csv* (bool): if set to *True*, the .csv file of the generated smiles will be exported in the specified directory.

*Notes*: 
- For the provided pre-trained models, only *fold=[1]* and *epoch=[9]* are provided.
- The list of available models and their description are provided in [evaluation/model_names.md](evaluation/model_names.md)

## Fine-tuning a model<a name="Finetuning"></a>

Fine-tuning requires a pre-trained model and a parameter file (.ini). 
Examples of the parameter files (BIMODAL and ForwardRNN) are provided in [experiments/](experiments/).

The fine-tuning set needs to be pre-processed, see [next section](#preprocessing).

You can start the sampling procedure with [model/main_fine_tuner.py](model/main_fine_tuner.py)


|Section		|Parameter     	| Description			|Comments |
| --- | --- | --- | --- |	
|Model		|model         	| Type				| ForwardRNN, BIMODAL  |
| 		|hidden_units	| Number of hidden units	|	Suggested value: 256 for ForwardRNN;  128 for BIMODAL|
|Data		|data		| Name of data file		| Has to be located in data/ |
| 		|encoding_size  | Number of different SMILES tokens		| 55 |
|		|molecular_size	| Length of string with padding	| See preprocessing |
|Training	|epochs		| Number of epochs		|  Suggested value: 10 |
|		|learning_rate	| Learning rate			|  Suggested value: 0.001|
|		|batch_size	| Batch size			|  Suggested value: 128  |
|Evaluation	| samples	| Number of generated SMILES after each epoch |  |
|		| temp		| Sampling temperature		| Suggested value: 0.7 |
|		| starting_token	| Starting token for sampling	| G |
|Fine-Tuning		|start_model         	| Name of pre-trained model to be used for fine-tuning				|   |

To fine-tune a model, you can run:

```
t = FineTuner(experiment_name = 'BIMODAL_random_512_FineTuning_template')
t.fine_tuning(stor_dir='../evaluation/', restart=False)
```

Parameters:
* *experiment_name*:  Name parameter file (.ini)
* *stor_dir*: Directory where outputs can be found
* *restart*: If True, automatic restart from saved models (e.g. to be used if your training was interrupted before completion)
   
Note:
-  The batch size should not exceed the number of SMILES that you have in your fine-tuning file (taking into account the data augmentation).

### Preprocessing <a name="preprocessing"></a>
Data can be processed by using [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py):
```
from main_preprocessor import preprocess_data
preprocess_data(filename_in='../data/chembl_smiles', model_type='BIMODAL', starting_point='fixed', augmentation=1)
```
Parameters:
* *filename_in* (str): name of the file containing the SMILES strings (.csv or .tar.xz)
* *model_type* (str): name of the chosen generative method
* *starting_point* (str): starting point type ('fixed' or 'random')
* *augmentation*(int): augmentation folds [Default = 1]

*Notes*:
* In [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py) you will find info regarding advanced options for pre-processing (e.g., stereochemistry, canonicalization, etc.)
* Please note that the pre-treated data will have to be stored in [data/](data/).

## Advanced functions <a name="advanced"></a>

If you want to personalize the pre-training or use advanced settings, please refer to the following repo: https://github.com/ETHmodlab/BIMODAL

## Authors<a name="Authors"></a>

**Authors of the provided code** (as in [this repo](https://github.com/ETHmodlab/BIMODAL))
* Robin Lingwood (https://github.com/robinlingwood)
* Francesca Grisoni (https://github.com/grisoniFr)
* Michael Moret (https://github.com/michael1788)

**Author of this tutorial** 
* Francesca Grisoni (https://github.com/grisoniFr)

See also the list of [contributors](https://github.com/ETHmodlab/Bidirectional_RNNs/contributors) who participated in this project.


## License<a name="License"></a>

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This code is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## How to Cite <a name="cite"></a>

If you use this code (or parts thereof), please cite it as:

```
@article{grisoni2020,
  title         = {Bidirectional Molecule Generation with Recurrent Neural Networks},
  author        = {Grisoni, Francesca and Moret, Michael and Lingwood, Robin and Schneider, Gisbert},
  journal       = {Journal of Chemical Information and Modeling},
  volume        = {60},
  number        = {3},
  pages         = {1175–1183}, 
  year          = {2020},
  doi           = {10.1021/acs.jcim.9b00943},
  url           = {https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943},
 publisher      = {ACS Publications}
}
```

```
@incollection{grisoni2021,
  author       = {Grisoni, Francesca and Schneider, Gisbert},
  title        = {De novo Molecule Design with Chemical Language Models},
  booktitle    = {Artfificial Intelligence in Drug Design},
  publisher    = {Springer},
  year         = 2021,
  volume       = {},
  series       = {Methods in Molecular Biology},
  chapter      = {},
  pages        = {201-213},
  address      = {The address of the publisher},
  edition      = {}
  }
```

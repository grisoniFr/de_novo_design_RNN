# Pre-trained models

This file contains information on the 20 pre-trained models that we provided.

## Table of Contents
1. [Architecture of the pre-trained models](#settings)
2. [Sampling from the pre-trained models](#sampling)

## Architecture of the pre-trained models<a name="settings"></a>

### Model architecture
The provided models were trained with the following architecture:
* **Forward RNN**, **NADE** and **FB-RNN**: Five layers (BatchNormalization, LSTM Layer 1, LSTM Layer 2, BatchNormalization, Linear).
* **BIMODAL**: seven layers (BatchNormalization, LSTM Layer 1 - forward, LSTM Layer 1 - backward, LSTM Layer 2 - forward, LSTM Layer 2 – backward, BatchNormalization, Linear).

A dropout value of 0.3 was used for the output weights in the first LSTM Layer. Models were trained with the Adam optimization algorithm, using cross-entropy loss for performance optimization, computed based on five-fold cross-validation (random partitioning protocol).
Models were trained for 10 epochs. Additional details are found in Tables 1-3.

**Table 1**. Details on the architecture of the Forward RNN and NADE.

|Type	| No. Units	|No. Parameters|
| ------ | ------ | ------ |
|BatchNormalization 1	|55	|110|
|LSTM 1	|256 or 512	|320512|
|LSTM 2	|256 or 512	|526336|
|BatchNormalization 2 |	256	|512|
|Linear Layer|	55	|14080|

**Table 2**. Details on the architecture of the FB-RNN models.

|Type	| No. Units	|No. Parameters|
| ------ | ------ | ------ |
|BatchNormalization 1|	110|	220|
|LSTM 1	|256 or 512	|376832|
|LSTM 2	|256 or 512	|526336|
|BatchNormalization 2 |	256	|512|
|Linear Layer|	55|	28160|


**Table 3**. Details on the architecture of the BIMODAL models.

|Type	| No. Units	|No. Parameters|
| ------ | ------ | ------ |
|BatchNormalization 1|	55|	110|
|LSTM 1 Forward	|128 or 256	|94720|
|LSTM 1| Backward	128 or 256|	94720|
|LSTM 2	|128 or 256|	132096|
|LSTM 2 Backward|	128 or 256	|132096|
|BatchNormalization 2 |	256|	512|
|Linear Layer	|55|	14080|



## Sampling from the pre-trained models<a name="sampling"></a>
The ID contained in the field "model name" can be use to sample from the pre-trained models, as explained in the README.

| *model name * | method |	starting point |	no. hidden |	augmentation
| ------ | ------ | ------ | ------ | ------ |
|'BIMODAL_fixed_1024'|	BIMODAL	|fixed|	1024|	none|
|'BIMODAL_fixed_512' |	BIMODAL	|fixed	|512|	none|
|'BIMODAL_random_1024'|	BIMODAL	|random	|1024|	none|
|'BIMODAL_random_1024_aug_5'	|BIMODAL|	random|	1024|	5-fold|
|'BIMODAL_random_512'	|BIMODAL	|random	|512|	none|
|'BIMODAL_random_512_aug_5'|	BIMODAL	|random	|512|	5-fold|
|'FBRNN_fixed_1024'	|FB-RNN	|fixed	|1024|	none|
|'FBRNN_fixed_512'	|FB-RNN	|fixed	|512	|none|
|'FBRNN_random_1024'	|FB-RNN	|random|	1024|	none|
|'FBRNN_random_1024_aug_5'|	FB-RNN	|random	|1024|	5-fold|
|'FBRNN_random_512'	|FBRNN|	random|	512	|none|
|'FBRNN_random_512_aug_5'|	FB-RNN|	random|	512|	5-fold|
|'ForwardRNN_1024'	|Forward RNN|	fixed|	1024|	none|
|'ForwardRNN_512'	|Forward RNN|	fixed|	512|	none|
|'NADE_fixed_1024'	|NADE|	fixed|	1024|	none|
|'NADE_fixed_512'	|NADE|	fixed|	512|	none|
|'NADE_random_1024'	|NADE|	random|	1024|	none|
|'NADE_random_1024_aug_5'|	NADE|	random|	1024|	5-fold|
|'NADE_random_512'|	NADE	|random	|512|	none|
|'NADE_random_512_aug_5'|	NADE|	random	|512|	5-fold|
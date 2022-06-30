In this folder:
	- The data set can be processed using the "extract_features_construct_dataset.py" script;
	- An unconstrained model can be trained using the "train_no_constraints.py" script;
	- A constrained model can be trained using the "train_constraints.py" script;
	- In "Constraints.py" file multiple constraints can be found for training constrained models;
	- In "attacks.py" two models can be attacked and their robustness compared.

The two models used in my thesis are provided in ~\Speaker recognition\bin, these can be used for testing.
The already processed data set is provided, so the training can start right away, it can be found in ~\Speaker recognition\RoDigits_splitV2.
The files and labels for the test data are saved in ~\Speaker recognition\test_dataset_to_add_noiseV2, it is used by the "attacks.py" script to add noise directly on top of audio.
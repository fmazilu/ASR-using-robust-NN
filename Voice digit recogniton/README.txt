In this folder:
	- The data set can be processed using the "extract_features_construct_dataset.py" script;
	- An unconstrained model can be trained using the "train_google_dataset.py" script;
	- A constrained model can be trained using the "train_constraints.py" script;
	- In "Constraints.py" file multiple constraints can be found for training constrained models;
	- In "attacks.py" two models can be attacked and their robustness compared.

The two models used in my thesis are provided in ~\Voice digit recogniton\bin, these can be used for testing.
The already processed data set is provided, so the training can start right away, it can be found in ~\Voice digit recogniton\processed_google_dataset.
The files and labels for the test data are saved in ~\Voice digit recogniton\test_dataset_to_add_noise, it is used by the "attacks.py" script to add noise directly on top of audio.
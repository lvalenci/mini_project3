# mini_project3
Repository for CS 155 Mini-Project 3  
Folders:  
- data: contains raw and processed data
    - the RNN models are labeled model_a_b_c.h5 where a = sequence offset, b = number of LSTM units, c = temperature 
- data_processing: contains single file, used to process Shakespeare's sonnets into desired form for both RNN and HMM models
- HMM_code: contains all python code for generating sonnets using HMMs. 
    - 10 Syllable Sonnets With Rhyming.ipynb: generates sonnets with rhyming and 10 syllables per line
    - 10_syllable_sonnets.ipynb: generates sonnets with 10 syllables per line
    - naive_sonnet.ipynb: generates a naive sonnet
    - HMM.py: HMM code from Set 6. Unmodified with exception that generate emissions has option of having seeding with a given word.
    - HMM_helper.py: HMM_helper code from Set 6 with significant alterations. All changes documented in the file itself using multiline comments (i.e. """comment"""). 
- RNN_code: contains code for generating sonnets using RNNs
    - RNNs.ipynb: trains recurrent neural networks using TensorFlow. Outputs models to data folder
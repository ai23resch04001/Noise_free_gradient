# Noise free gradients
#step 1: install the necessary python packages , preferably with same version as mentioned in requirements.txt

`pip3 install -r requirements.txt`

#step 2: Download the tiny imagenet dataset and put it in scripts/data folder.

#step 3: run the model training program 

`cd scripts`
`./train_for_selection.sh`



#step 4: gradient computation, similarity score computation and nearest neighbor determination

`cd scripts`
`./calculate_similarity.sh`

#step 5: train the model with coreset 

#to understand various parameters of the program, run `python3 coreset_train.py --help`


`cd scripts`
`./run_coreset_training.sh 0.2 0.05`

#the above command will run coreset selection on 5% of dataset with a batch size of 128 and initial learning rate 0.01 for 10 iterations , with 200 epochs each with a selection threshold of 0.2. Test accuracy will be calculated at 2 epoch intervals. 

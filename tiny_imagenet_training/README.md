# Noise free gradients
#step 1: install the necessary python packages , preferably with same version as mentioned in requirements.txt

`pip3 install -r requirements.txt`

#step 2: Download the tiny imagenet dataset and put it in scripts/data folder

#step 2: run the model training program 

`python3 -W ignore train_and_save_models.py`

#step 3: plot the loss vs epoch plot

`python3 plot_training_loss.py`

#select a number till which loss is decreasing consistently as the number of models to be used. From the attached sample plot, we have chosen 7. 

#step 4: create mapping data 

`python3 map_images.py` 

#it will generate a pickle file named mapping_data.pkl 

#step 5: gradient computation, similarity score computation and nearest neighbor determination

`python3 calculate_gradients.py --number 7`

#step 6: train the model with coreset 

#to understand various parameters of the program, run `python3 coreset_CIFAR100_resnet18.py --help`

#one example :

`python3 coreset_CIFAR100_resnet18.py --input 7 --number 5 --decimal 1 --batch_size 128 --lr 0.01 --weight_decay 0.0005 --iterations 5 --epochs 200 --test_interval 5`

#the above command will run coreset selection on 0.5% of dataset with a batch size of 128 and initial learning rate 0.01 for 5 iterations , with 200 epochs each. Test accuracy will be calculated at 5 epoch intervals. 

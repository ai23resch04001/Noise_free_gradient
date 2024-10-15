for radius in $1
do 
    for fraction in $2
    do
        python3 -W ignore ../coreset_train.py --epochs 200 --dataset tiny --batch_size 128 --model resnet18 --se 10 --device cuda:0 --lr 0.1 --weight_decay 5e-4 --fraction $fraction --optimizer SGD --scheduler cosine --iterations 10 --random_seed 1 --test_interval 2 --test_eval_start_epoch 100 --wandb 0 --radius $radius
    done 
done 

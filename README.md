# Pull the repo

## Create a Virtual Environment and kick it off

## Install the required libraries already listed in requirements.txt

    - pip install -r requirements.txt ## (windows)

## Already have the FFNN.py and RNN.py ran and saved the models.
    - rnn_stats_h64_e6.pkl
    - ffnn_model_h50_e10.pkl

## But you can still run it on different parameters like hidden layers=70 and epoch=15 etc.

Code to run RNN.py: python rnn.py --hidden_dim 64 --epochs 10 --train_data training.json --val_data validation.json

Code to run FFNN.py: python ffnn.py --hidden_dim 50 --epochs 10 --train_data training.json --val_data validation.json

## After the models are trained you can directly run the file visualize_results.py which contains the code for model comparisons and analysis. (This will save the analysis images in the root folder)

## Do not forget to change the name of the models in visualize_results.py.

![alt text](image.png)
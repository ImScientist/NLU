A model applied to the [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge) dataset.

## Theory

- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [Attention is all you need (step by step explanation)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [BERT](https://arxiv.org/pdf/1810.04805.pdf)

## Quick start

- Download the data from [here](https://www.kaggle.com/c/google-quest-challenge/overview), modify the 
  variables `DATA_DIR`, `RESULTS_DIR` from .env and load it:  
    ```bash
    source .env
    ```
    create a python virtual environment and install the dependencies:
    ```bash 
    conda create nlu python=3.6
    conda activate nlu 
    pip install -r requirements.txt
    python setup.py install
    ```

- Train the model (remove `size_tr_val` to use the complete dataset; `size_val` refers to the size of the validation
 dataset): 
    ```bash 
    python exec/train_google_qa.py \
        --data_path ${DATA_DIR}/train.csv \
        --model_dir ${RESULTS_DIR}/models \
        --log_dir ${RESULTS_DIR}/logs \            
        --size_tr_val 100\
        --size_val 40\
        --batch_size 6 \
        --num_epochs 2 \
        --print_freq 10 \
        --seed 10
    ```

- Make a prediction (only for the first 100 elements from the test set):
    ```bash 
    python exec/predict_google_qa.py \
        --data_path ${DATA_DIR}/test.csv \
        --result_dir ${RESULTS_DIR}/results \
        --model_dir ${RESULTS_DIR}/models \
        --load_epoch 1 \
        --batch_size 2 \
        --n_el 100
    ```

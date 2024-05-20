# Training NLP (Jigsaw Religion) Data with Anonfair

This repository contains the Anonfair tool for running and analyzing the Jigsaw religion data. The data includes three religions: Christian, Muslim, and other.

## Installation

Follow these steps to set up the environment:

1. **Create a new conda environment:**
    ```sh
    conda create -n anonfair python=3.8
    ```

2. **Install required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Activate the conda environment:**
    ```sh
    conda activate anonfair
    ```

## Download Dataset

Run the script to download the dataset:

```sh
./download_data.sh
```

## Run the Code

Execute the training script:

```sh
./train.sh
```

## Analyze the Results with Anonfair

To analyze the results, go to the `examples` folder and open the notebook `training_nlp_jigsaw_religion.ipynb`:

1. Navigate to the `examples` directory:
    ```sh
    cd ..
    ```

2. Open the Jupyter Notebook:
    ```sh
    jupyter notebook training_nlp_jigsaw_religion.ipynb
    ```

This notebook provides detailed analysis with Anonfair of the results obtained from training and testing the model on the Jigsaw religion data.



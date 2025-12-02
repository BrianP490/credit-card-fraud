# 💳 CREDIT CARD FRAUD PREDICTER

## 📌 Problem Statement

As credit fraud is a reality we all face today, this project helps identify transactions made without the consent of the card owner. 

## 🔎 Scope

The scope of the project is to make an agent that is useful in classifying if a transaction was fraudulent or not. The agent can be used for an application running queries from a front-end system or implemented as an agent in a multi-agent application.

## 📖 Project Purpose
This project is useful in understanding classification problems with this case only having to identify between 2 classes. After understanding the process for 2-class classification, the same procedure can be scaling up to a much wider multi-class classification.

## 💻 Installation Instructions:

1. Make sure you have [Conda](https://conda.org/) installed which is the virtual environment and package manager system used.

2. Clone the repository

    ```bash
    git clone INSERT-GITHUB-URL-HERE
    ```

    ```bash
    # Example
    C:\Users\brianperez\Desktop\DEV>git clone https://github.com/BrianP490/fake-repository.git
    ```

3. Create the conda environment for the dependencies 
    - More information about the virtual environments creation & activation located in the ['./setup/README.md'](./setup/README.md)

## 🏋️ Training Script Guide

<details>

Complete configuration file documentation found in: [`./configs/README.md`](./configs/README.md)

However, these are the most important command-line arguments that affect the behavior of the training script:

- --epochs

    - Type: int

    - Default: `8`

    - Description: Number of training epochs to run.

- --learning_rate

    - Type: float

    - Default: `0.0003`

    - Description: Learning rate used by the optimizer.

- --max_grad_norm

    - Type: float

    - Default: `3.0`

    - Description: The Maximum L2 Norm of the gradients for Gradient Clipping.

- --dataloader_batch_size

    - Type: int

    - Default: `64`

    - Description: Batch size used by the dataloaders for training, validation, and testing.

- --dataloader_pin_memory

    - Type: action='store_true' (boolean flag)

    - Default: `false (if flag is not present)`

    - Description: Toggle pinned memory in dataloaders (disabled by default). Include this flag to enable it.

- --dataloader_num_workers

    - Type: int

    - Default: `0`

    - Description: Number of subprocesses to use for data loading.

- --log_iterations

    - Type: int

    - Default: `2`

    - Description: Frequency (in iterations) to log training progress.

- --eval_iterations

    - Type: int

    - Default: `2`

    - Description: Frequency (in iterations) to evaluate the model.

- --use_cuda

    - Type: action='store_true' (boolean flag)

    - Default: `false (if flag is not present)`

    - Description: Toggle CUDA for training if available. Include this flag to enable CUDA.

- --device

    - Type: str

    - Default: `"cpu"`

    - Description: Device to use for training (e.g., "cpu", "cuda:0"). This parameter overrides the --use_cuda flag if specified.

- --save_model

    - Type: action='store_true' (boolean flag)

    - Default: `false (if flag is not present)`

    - Description: Toggle saving the trained model after training. Include this flag to enable model saving.

- --model_output_path

    - Type: str

    - Default: `"./models/trained-model.pt"`

    - Description: File path to save the trained model. Parent directories will be created if they do not exist.

### 📝 Example Commands:

- ```python train.py```
    - Uses the default settings to run the script.
    
- ```python train.py --epochs=32  --log_iterations=4 --eval_iterations=8 --save_model```
    - Explanation: Run for 32 epochs and log the average batch iteration Mean Absolute Error Loss every 4 iterations. Evaluate the Policy under training every 8 epochs. Lastly, save the trained model. Uses the default save path. Uses default for everything else.
    
    
- ```python train.py --epochs=32  --use_cuda --save_model --model_output_path=models/first-trained-model.pt```
    - Explanation: Let the system detect if your system has GPU capabilities and has cuda available for training. During the training setup, the system dynamically sets the device variable for model training. Save the trained model using the specified location. Uses default for everything else.

</details>

## 🗠 Dataset Information

 - All dataset information located in [HuggingFace Dataset](https://huggingface.co/datasets/MaxPrestige/credit-card-fraud-CLEAN)

## 📂 Folder Directory

 * Created using [githubtree](https://github.com/rescenic/githubtree)

```bash
# credit-card-fraud

setup/
    └── requirements.yaml
.gitignore
README.md
```
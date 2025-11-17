# Environments used:

1. `CC_env`
    - Requirements file: **requirements.yaml**
    - Used during Model Development and Training.

2. `strl_env`
    - Requirements file: **streamlit_dev.yaml**
    - Used during Streamlit Application Development

# Environment Creation:

1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

2. Open a new terminal and navigate to the cloned directory

```bash
# Example
C:\Users\brianperez\Desktop\DEV>cd cloned-folder
C:\Users\brianperez\Desktop\DEV\cloned-folder>
```

3. Create the environment using the Requirements File

```bash
# Example
C:\Users\brianperez\Desktop\DEV\cloned-folder>conda env create -f ./setup/fake-requirements.yaml
```

4. Follow through with the installation pressing "y" when prompted.

# Environment Activation:

1. Use: "conda activate *Name-of-Env*" 

```bash
# Example
C:\Users\brianperez\Desktop\DEV\cloned-folder>conda activate Fake-Env
```

2. You should see the name of the environment prepended to your current working directory in the terminal.

```bash
# Example
(Fake-Env) C:\Users\brianperez\Desktop\DEV>
```

### Note for running any of the Interactive Python Notebooks:

 - Select "install ikernel" when popup appears when first activating running a cell in an .ipynb file with the env

# Environment Deactivation:

1. Use: "conda deactivate"

```bash
# Example
(Fake-Env) C:\Users\brianperez\Desktop\DEV>conda deactivate
C:\Users\brianperez\Desktop\DEV>
```

## For more information on conda visit:
 - [GETTING STARTED WITH CONDA](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
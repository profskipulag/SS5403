# ST540203 train.py
A package for generating emulators, or fast forward models. These are functions that adequatrely approximate the output of a physics based model such as Fall3D, for a given set of inputs, but perform the calculation at much faster speeds, facilitating their insertion into Monte Carlo Bayesian frameworks. The package consists of `pyem`, a package that contains some helper classes, including

 * `Emulator`, a class for creating emulators, currently limited to simple interpolate-and-sum "puff" based emulators that leverage the linear properties of some models, in this case basic Fall3D runs.

The useage of `Emulator` is illustarted in the Jupyter notebook `notebook.ipynb`, and the script `train.py` is the application of the library for the purposes of DTC4. 

## To do
 * Neural network based emulators for models that are nonlinear in their relationship between flux and ground concentration
 * fill in train.py
 
## Package structure


    ST540203/
    ├── environment.yaml      - configures conda environment with required packages
    ├── LICENSE               - GPL 3
    ├── notebook.ipynb        - example use of the pywaf package
    ├── pyem                  - python emulator package
    │   ├── __init__.py       - initialises the package
    │   └── source.py         - source code for various classes
    ├── README.md             - this file
    ├── .gitignore            - files to be ignored by git
    └── train.py              - script that calls the package for DTC4


## To download the repository
Clone the repository to your machine

    git clone https://github.com/profskipulag/ST540203.git

You will be asked for your username and password. For the password github now requires a token:
- on github, click yur user icon in the top right corner
- settings -> developer settings -> personal access tokens -> Tokens (classic) -> Generate new token -> Generate new token (classic) 
- enter you authentifcation code
- under note give it a name, click "repo" to select al check boxes, then click generate token
- copy result enter it as password

## To run the jupyter notebook
Create a new conda environment from the environment.yaml file:

    conda env create -f environment.yaml

Activate the environment

    conda activate st540203
    
Launch the notebook server

    jupyter notebook
    
Navigate to the st540203 directory and click the file `notebook.ipynb` to launch it.
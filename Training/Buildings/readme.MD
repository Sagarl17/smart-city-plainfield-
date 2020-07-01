# Buildings Training

This folder contains code necessary to train for the building classification model .

# Steps For Installation:

```bash
git clone https://git.xyzinnotech.com/gopinath/smart-city-plainfeild.git
cd smart-city-plainfeild
For linux users,download and install miniconda from following link:
    https://repo.anaconda.com/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh

For windows users,download and install miniconda from following link:
    https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Windows-x86_64.exe

Open terminal/Command prompt inside the folder:
If you are installing on a system with GPU access:
    conda env create -n plainfield -f environment-gpu.yml
otherwise:
    conda env create -n plainfield -f environment.yml

conda activate plainfield
conda install -c pytorch -c fastai fastai
conda install jupyter
    


    
```

# Training Buildings:
```bash
Please download the files from the following links and place it in the folder.
https://drive.google.com/file/d/1c9vK2mkIJbTa317pI8GfTJAvEoD6B0yP/view?usp=sharing
https://drive.google.com/file/d/1Js9qYMRTzs-GkBSbBkdaAmIq9uSKI7jX/view?usp=sharing
https://drive.google.com/file/d/1S-Fxc0uhqOhd7cNH52UPBUepm6aZ1HSS/view?usp=sharing

Create a dataset folder inside the "data" folder and place the following data inside the dataset folder.

Open terminal in folder and run jupyter notebook.
Open Create_training_dataset.ipynb and run the cells to create the necessary data.
Open segmenation.ipynb and make necessary file path changes as per your folder locations and run the cells.
Change the weights in unet_learner as per your requirements. Decide the learning rat and the epochs as needed. 



```


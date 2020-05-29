# Smart City Plainfield

This repository contains the code for extracting jsons of trees,buildings and roads of US cities in winter.

# Steps For Installation:

```bash
git clone https://git.xyzinnotech.com/gopinath/smart-city-plainfeild.git
cd smart-city-plainfeild
For linux users,download and install miniconda from following link:
    https://repo.anaconda.com/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh

For windows users,download and install miniconda from following link:
    https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Windows-x86_64.exe

If you are installing on a system with GPU access:
    conda env create -n plainfield -f environment-gpu.yml
otherwise:
    conda env create -n plainfield -f environment.yml
    
```

# Data Preparation:
```bash
Please download the models from the following links and place it in models folder.
https://drive.google.com/open?id=1SQ0FSsctDaXvKhAb01e4K5Xb8dZh2iH9
https://drive.google.com/open?id=1Ltinw3oyvpMzmmlyFESDvPKK3mp9kWCK
https://drive.google.com/open?id=1-4E0y3ElzTPsL_Ku9NwRBUs_VYxYk_oi

Create a dataset folder inside the "data" folder and place the following data inside the dataset folder
    *  Orthomosiac file of the city
    *  DTM file of city which includes dtm in filename
    *  DSM file of city which includes dsm in filename
    *  Pointcloud of city in '.las' format
```
# How to test for the new dataset:

```bash
python main.py "dataset_folder_name" trees roads buildings

If the dataset folder is named 'city1',then it should be run in following way:

    python main.py city1 trees roads buildings

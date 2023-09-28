# EuropeanResearchersNight2023

Scripts for the EOmaps widget to visualize the Parkistan-flood at the ERN 2023

To setup from scratch, the following steps are required:

- install [miniconda]() 

- install all required dependencies via:
  
  ```
  conda install -c conda-forge mamba
  mamba create -n ern23 -c conda-forge eomaps=7.1.2 rioxarray
  ```

Once the installation is complete, **activate** the environment with 

```
conda activate ern23
```

then navigate to the directory where you cloned the repository and run:

```
python -m widget_TIFF.py
```
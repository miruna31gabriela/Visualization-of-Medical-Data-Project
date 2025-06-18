# Radiomic Visual Representation Tool for Breast Tumors​
> Authors: Miruna Vasile & Andreea Pavel

> Credits: Data source from The Cancer Institute Archive [NH16]; 

This project pesents a visualization tool protoype for breast tumors along with 
corresponding radiomic features and their evolution over time.

--- 

## Description 

The tool enables exploratory analysis and clinical interpretation by providing:
- Interactive 3D visualization of tumor volumes across treatment stages;
- Temporal tracking of radiomic features to observe changes over time and assess
  treatment response at the individual or patient cohort level.


## Installation / Requirements
Python 3.8.0 was used for this project. The packages that are need can be installed as follows in the command prompt:

```bash
py -3.8 -m venv name_env # name of environment
name_env\Scripts\activate
python -m pip install --upgrade pip

pip install os
pip install pathlib
pip install typing
pip install vtk
pip install pydicom
pip install numpy
pip install matplotlib
pip install pandas
pip install tkinter
pip install logging
pip install pyradiomics
```

For pyradiomics the installation process is the same, but an error may be raised. If that error appears, do the following:
- Download from this repository the *.whl file;
- In the command prompt, write ```bash pip install path_of_file```;
- run ```bash pip install pyradiomics``` again and it should work.


## How to run


## Features 



## References
[NH16] Newitt D., Hylton N. *Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy*. The Cancer Imaging Archive Version 3 (2016). [https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy](https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy)


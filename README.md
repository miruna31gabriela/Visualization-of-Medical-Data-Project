# Radiomic Visual Representation Tool for Breast Tumors​
> Authors: Miruna Vasile (TU Wien, e12442832) & Andreea Pavel (TU Wien, e12450451)

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

pip install pathlib
pip install typing
pip install vtk
pip install pydicom
pip install pandas
pip install tk
pip install logging
pip install pyradiomics
```

For `pyradiomics` the installation process is the same, but an error may be raised.

**NOTE**: This error originates from a subprocess, and is likely not a problem with `pip`. 
```bash
ERROR: Failed building wheel for SimpleITK Failed to build SimpleITK
ERROR: Failed to build installable wheels for some pyproject.toml based projects (SimpleITK)
```

If that error appears, do the following steps:
- Download from this repository the *.whl file; This file was obtained from [https://github.com/SimpleITK/SimpleITK/releases/tag/v2.1.1](https://github.com/SimpleITK/SimpleITK/releases/tag/v2.1.1). **NOTE**: the *.whl file we provided works for Windows; if you need the file for Linux or MacOS, they can be found in the provided GitHub repository.
- In the command prompt, write `pip install path_of_file # file path name`
- run `pip install pyradiomics` again

## How to run
After installing all the packages, select the previously created virtual environment in the editor of your choosing. 


## References
[NH16] Newitt D., Hylton N. *Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy*. The Cancer Imaging Archive Version 3 (2016). [https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy](https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy)


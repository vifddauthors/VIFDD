# VIFDD: Visual Intrusion and Fraud Detection Dataset

## Overview

The Visual Intrusion and Fraud Detection Dataset (VIFDD-2024) is a comprehensive collection of visual data designed for research and development in the field of intrusion and fraud detection in web-based environments. This dataset includes a variety of normal and fraudulent visual samples gathered from diverse online sources.

## Directory Structure

```
VIFDD/
|-- src/
|   |-- main.py
|   |-- requirements.txt
|-- datacollection/
|   |-- screenshot.js
|   |-- fradulentlist.csv
|   |-- websites.csv
|   |-- batch_csv.py
|-- samples/
|   |-- normal/
|       |-- images...
|   |-- fraudulent/
|       |-- images...
|-- dataset.txt
```

## Peek into the dataset
![NEUR IPS 2024-VIFDD data viz](https://github.com/mldlusinkaggle/VIFDD/assets/171614320/38b0fd7a-d157-4645-8d24-6fde23d0f755)



## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mldlusinkaggle/VIFDD
   cd VIFDD-2024
   ```

2. Install required packages:
   ```bash
   pip install -r src/requirements.txt
   ```

3. Run benchmarking:
   ```bash
   python src/main.py --dataset_path path/to/dataset
   ```

## Dataset Link

The complete dataset can be accessed through the link provided in `dataset.txt`.

## License

Please review the LICENSE file for details regarding the usage and distribution of the dataset.

## Acknowledgments

If you use this dataset in your research, please consider citing this repository.

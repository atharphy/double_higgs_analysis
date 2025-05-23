# Higgs Pair vs tt̄ Classification Pipeline

This repository provides a pipeline to classify events from Higgs boson pair production (HH) and top quark pair production (tt̄) using a neural network.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/atharphy/double_higgs_analysis.git
   cd double_higgs_analysis
   ```

2. Download the following CSV files:
   - hh.csv from: https://drive.google.com/file/d/1dPQpChRBzXiG0yTrvx5a6WD2qm4h61io/view?usp=sharing
   - tt.csv from: https://drive.google.com/file/d/1DREyeYmgJ6Egde_C1u8cGmfSUuAMmgmK/view?usp=drive_link

3. Place both CSV files in the root directory of the repository (the same location as the notebooks).

4. Ensure that Python and necessary packages are installed. If using a virtual environment:
   ```
   python -m venv env
   source env/bin/activate     # On Windows: env\Scripts\activate
   ```

5. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

6. Run the following files in the order listed below. Each step depends on the successful execution of the previous one.

   1. preprocess.ipynb
   2. train.ipynb
   3. classify.ipynb
   4. analyze.ipynb

## File Descriptions

### preprocess.ipynb

This notebook performs the data preprocessing steps. It reads in the hh.csv and tt.csv files, assigns appropriate class labels to the signal and background events, and combines them into a single dataset. It then performs checks on the data, applies any necessary transformations such as normalization, The resulting processed datasets are saved in the appropriate formats to be used in the training step.

### train.ipynb

This notebook loads the preprocessed training and validation datasets and defines a neural network model. The model is compiled with relevant loss functions and optimization algorithms. It is then trained over several epochs, with metrics such as accuracy and loss being tracked. The trained model is saved to disk, and plots showing the training and validation performance over time are generated and stored.

### classify.ipynb

This notebook loads the saved model from the training step and applies it to the test data to perform inference. It calculates the predicted outputs (such as probabilities or labels) for each event in the test set. These predictions are saved for further analysis. This notebook may also include some preliminary evaluation steps, such as generating confusion matrices or simple classification statistics.

### analyze.ipynb

This notebook loads the predictions made in the classify step and calculates the significance (sigma value) for the HH vs tt̄ classification. The results provide a statistical measure of how well the model distinguishes between the two classes. Visualizations or summary statistics may be generated as part of this step depending on the analysis logic.

### visualize.py

This Python script contains helper functions for generating plots used during training and analysis. It includes:

- A function to draw the ROC curve and compute the AUC.
- A function to visualize training accuracy and loss over epochs.
- A function to plot the neural network discriminant for signal and background events, comparing training and test sets.

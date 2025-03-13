# HydroMLP
## PCA-Integrated Multi-Layer Perceptron Neural Network for Predicting Reverse Osmosis Membrane Fouling

### Overview
Membrane fouling remains a significant challenge in Reverse Osmosis (RO) water treatment systems, reducing efficiency and increasing maintenance costs. This project develops a PCA-integrated Multi-Layer Perceptron (MLP) neural network to predict membrane fouling by leveraging historical operational data. The model applies Principal Component Analysis (PCA) to reduce feature dimensionality and a deep learning approach to forecast Stage 3 flux, aiding in proactive maintenance planning.

<img width="1104" alt="image" src="https://github.com/user-attachments/assets/cebc475b-0125-49cc-ae1d-0918ceee91a0" />

## Problem Statement
RO systems are widely used in water treatment, but membrane fouling reduces efficiency and increases operational costs. Existing predictive models struggle to generalize across varying operational conditions. This project employs machine learning to improve membrane fouling prediction accuracy.

## Approach
### Data Collection
- Historical operational data from A01 RO unit.

### Preprocessing
- Outlier removal (Hampel filter)
- Smoothing (moving average)
- Standardization

### Feature Engineering
- Dimensionality reduction via PCA.

### MLP Model
- Neural network with ReLU activation, trained using the Adam optimizer.

### Evaluation
- RMSE and R² used for model assessment.

## Installation and Setup
### Clone the Repository
```bash
git clone https://github.com/Greatestapple/HydroMLP.git
cd HydroMLP/Data
```

### Install Required Dependencies
Install required libraries:
```bash
pip install numpy pandas matplotlib scipy tensorflow scikit-learn
```

## Data Preparation
### Downloading the Data
The dataset consists of historical operational data from multiple RO units. Place the dataset files in the `Data/` directory. If using external data, ensure the files are in CSV format with the required fields:
```
date (timestamp)
feed_pressure (psi)
feed_flow
turbidity (NTU)
pH
temperature (°F)
TOC (mg/L)
cleaned (binary: 0/1)
dsc (days since cleaning)
dsr (days since replacement)
stage3_flux (target variable)
```

### Preprocessing Steps
**Outlier Removal:** Hampel filter replaces outliers using median absolute deviation:
```math
\tilde{x}_i = \begin{cases} x_i, & |x_i - \tilde{x}_{med}| \leq n \cdot \sigma_{MAD} \\ \tilde{x}_{med}, & |x_i - \tilde{x}_{med}| > n \cdot \sigma_{MAD} \end{cases}
```

**Smoothing:** Moving average with a window size of 10.

**Feature Scaling:** Standardization using:
```math
X_{scaled} = \frac{X - \mu}{\sigma}
```

## Running the Model
### Execute the Main Script
```bash
python HydroMLP.py
```

### Expected Outputs
- Preprocessed data visualization.
- PCA variance plot.
- Prediction vs. actual scatter plot.
- Model performance metrics (RMSE, R²).

## Model Architecture
- **Input Layer:** PCA-transformed features.
- **Hidden Layers:** Three layers with 25, 25, and 40 neurons, respectively.
- **Activation Function:** ReLU.
- **Output Layer:** Predicting Stage 3 flux.
- **Loss Function:** Mean Squared Error (MSE):
  ```math
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  ```
- **Optimizer:** Adam.

## Evaluation Metrics
Model performance is assessed using:

**Root Mean Squared Error (RMSE):**
```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

**Coefficient of Determination (R²):**
```math
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
```

## Repository Structure
```
HydroMLP/
│-- Data/         # Contains RO system operational data
│-- HydroMLP.py   # Main execution script
│-- README.md     # Project documentation
```

## Contributors
- **Jiayi Chen** - [cjiayi@stanford.edu](mailto:cjiayi@stanford.edu)
- **Tianchen He** - [hetc@stanford.edu](mailto:hetc@stanford.edu)
- **Xinyu Jing** - [xj1225@stanford.edu](mailto:xj1225@stanford.edu)

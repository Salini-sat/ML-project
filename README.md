# Income Segmentation Using K-Means Clustering

## Overview
This project implements an **unsupervised machine learning model** to perform income-based customer segmentation using **K-Means clustering**. The goal is to identify distinct user groups based on age and income patterns, which can be useful for targeted analysis and decision-making.

## Problem Statement
Understanding customer segments is essential for applications such as personalized services, financial planning, and market analysis. This project groups individuals into meaningful clusters using age and income data without predefined labels.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Key Features
- Data preprocessing and normalization using **MinMaxScaler**
- Cluster formation using **K-Means algorithm**
- Optimal cluster selection using the **Elbow Method**
- Visualization of clusters and centroids
- Export of clustered data for further analysis

## Dataset
The dataset contains the following attributes:
- **Age**
- **Annual Income**

All features are scaled to ensure accurate distance-based clustering.

## Implementation Details
- Feature scaling is applied to avoid bias due to differing data ranges.
- K-Means clustering is performed with a fixed random state for reproducibility.
- The elbow method is used to determine the optimal number of clusters.
- Cluster assignments are added to the dataset and saved as a CSV file.

## How to Run the Project
1. Clone the repository:
2. Install required dependencies:
3. Run the script:

## Output
- Cluster visualization plot showing segmented groups
- `clustered_income.csv` file containing cluster labels for each data point

## Applications
- Customer segmentation
- Income-based market analysis
- Behavioral pattern identification
- Entry-level machine learning experimentation

## Future Enhancements
- Add more features for improved clustering
- Compare clustering algorithms (DBSCAN, Hierarchical)
- Deploy as a web-based visualization tool

## Author
Salini Satpathy  
GitHub: https://github.com/Salini-sat

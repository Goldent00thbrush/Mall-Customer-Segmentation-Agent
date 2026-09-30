#  Mall Customer Segmentation Agent

An unsupervised machine learning project that groups mall customers into behavioral segments using **K-Means clustering**, then maps each segment to a customer persona and recommended marketing action through an interactive Gradio interface.

The project uses customer demographic and spending information to identify groups of customers with similar characteristics. A trained K-Means model is then used to classify new customer profiles into one of the discovered segments.

<img width="1894" height="900" alt="Recording 2026-09-30 195938" src="https://github.com/user-attachments/assets/ce0d5943-469e-44a5-a735-6c19f23b1197" />

##  Project Overview

The project follows this workflow:

```text
Mall Customer Dataset
          ↓
     Data Cleaning
          ↓
   Gender Encoding
          ↓
    Feature Scaling
          ↓
   K-Means Analysis
          ↓
 Choosing Number of Clusters
          ↓
    Customer Segments
          ↓
   Segment Personas
          ↓
   Marketing Actions
          ↓
   Gradio Interface
```

##  Objectives

* Explore customer characteristics using unsupervised learning.
* Segment customers according to demographic and spending behavior.
* Determine a suitable number of clusters using clustering evaluation techniques.
* Analyze the characteristics of each customer segment.
* Assign descriptive personas to the resulting clusters.
* Provide recommended marketing actions for each segment.
* Build an interactive interface for assigning new customers to a segment.

##  Dataset

The project uses the **Mall Customers** dataset.

The dataset contains customer information including:

* Customer ID
* Gender
* Age
* Annual Income (`k$`)
* Spending Score (`1-100`)

`CustomerID` is removed because it is an identifier rather than a behavioral feature.

##  Data Preprocessing

The notebook performs the following steps:

1. Downloads the dataset using `kagglehub`.
2. Loads `Mall_Customers.csv`.
3. Removes `CustomerID`.
4. Converts `Gender` into a numerical representation using one-hot encoding.
5. Converts `Gender_Male` to an integer representation.
6. Standardizes the features using `StandardScaler`.

The clustering features are:

```text
Age
Annual Income (k$)
Spending Score (1-100)
Gender_Male
```

Scaling is important because the features have different numerical ranges.

##  Choosing the Number of Clusters

The notebook explores different values of `K` using two techniques.

### Elbow Method

K-Means models are evaluated for:

```text
K = 1 through 10
```

The notebook records the model's inertia and plots an elbow curve to help identify an appropriate number of clusters.

### Silhouette Score

Silhouette scores are calculated for:

```text
K = 2 through 10
```

The scores provide another way of examining how well-separated the resulting clusters are.

## K-Means Model

The final clustering model uses:

```python
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

The resulting cluster labels are added to the original, unscaled DataFrame.

## Cluster Summary

The notebook calculates the average characteristics and customer count for each cluster.

| Cluster | Avg. Age | Avg. Income (k$) | Avg. Spending Score | Customers |
| ------: | -------: | ---------------: | ------------------: | --------: |
|       0 |    32.69 |            86.54 |               82.13 |        39 |
|       1 |    36.48 |            89.52 |               18.00 |        29 |
|       2 |    49.81 |            49.23 |               40.07 |        43 |
|       3 |    24.91 |            39.72 |               61.20 |        54 |
|       4 |    55.71 |            53.69 |               36.77 |        35 |

The notebook also visualizes the resulting segments using annual income and spending score.

##  Customer Personas

The five clusters are mapped to descriptive personas in the application:

| Cluster | Persona                      | Recommended Action                                           |
| ------: | ---------------------------- | ------------------------------------------------------------ |
|       0 | Sensible / Budget Shoppers   | Value bundles and discount coupons for essential items       |
|       1 | Careless / Impulse Spenders  | Trend notifications and low-installment payment options      |
|       2 | Standard Shoppers            | General weekly catalog and store updates                     |
|       3 | Careful / High-Income Savers | Premium-brand and quality-focused messaging                  |
|       4 | VIP / High-Value Customers   | VIP access, early product drops, and personal shopper alerts |

These personas and actions are manually defined in the notebook after the clusters are created.

##  Segmentation Agent

The Gradio application allows a user to enter a new customer profile.

Inputs include:

* Customer ID / Name
* Gender
* Age
* Annual Income
* Spending Score

The application:

1. Converts the user input into the same feature structure used during training.
2. Applies the trained scaler.
3. Uses the trained K-Means model to predict the customer's cluster.
4. Looks up the corresponding persona.
5. Returns the recommended marketing action.

The output contains:

```text
Customer ID
Assigned Cluster
Persona
Recommended Marketing Action
```

##  Interactive Gradio Application

The interface is titled:

**Mall Customer Segmentation Agent**

The application allows customer attributes to be adjusted interactively so that different customer profiles can be tested against the trained segmentation model.

The interface uses:

```python
interface.launch(share=True, debug=True)
```

which can create a temporary public Gradio sharing URL when run in a compatible environment.

##  Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* KaggleHub
* Gradio

## Project Structure

A typical project structure can be:

```text
.
├── Unsupervised_Learning.ipynb
└── README.md
```

The clustering model and scaler are currently created within the notebook rather than saved as separate model files.

## Running the Project

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub gradio
```

Then open:

```text
Unsupervised_Learning.ipynb
```

Run the notebook cells in order.

The final cell launches the interactive Gradio application.

Or use link https://colab.research.google.com/drive/1hPO6zYbjFme5PL4dXGE4uTADZ4DNccP4?authuser=1#scrollTo=64FWMactlW4F

## Notes

This project is a demonstration of customer segmentation using unsupervised learning.

K-Means cluster numbers themselves do not have an inherent business meaning. The descriptive personas are assigned afterward in the notebook based on the characteristics of the resulting clusters.

The recommended marketing actions are demonstration recommendations defined in the `CLUSTER_PERSONAS` dictionary; they are not connected to a live marketing platform.

## Possible Extensions

Potential improvements include:

* Save the trained K-Means model and scaler with Joblib.
* Build a reusable preprocessing pipeline.
* Add automated cluster profiling.
* Add interactive cluster visualizations to the Gradio application.
* Compare K-Means with other clustering algorithms.
* Add additional customer behavioral features.
* Generate automated explanations for why a customer belongs to a particular segment.
* Connect segment assignments to a real marketing platform.
* Add model/version tracking for updated customer segmentation.




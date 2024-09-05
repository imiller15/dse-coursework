Deep Learning for Network Intrusion Detection System
==============================

In this project we performed experiments on the (UNSW-NB15  dataset)[https://research.unsw.edu.au/projects/unsw-nb15-dataset] to create intrusion detection machine learning models for both binary and multiclass classification. Our approach consists of three stages: data preprocessing and feature engineering, baseline machine learning methods for which we selected logistic regression, random forest and k-nearest neighbors, and deep learning methods. The experimental results show that the task of identifying an event of an attack happening could be solved by classical machine learning tools, achieving an accuracy over 99% without a need to employ neural networks. However, the problem of correctly identifying a specific type of attack has proven to be a lot more challenging and the highest balanced accuracy achieved with a deep learning classifier has only reached 77.24%.

Project Organization
------------


    ├── README.md          <- The top-level README for this project.
    │
    ├── docs               <- Project proposal.
    │   └── papers         <- Annotated papers related to the project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0_JQP_EDA`.
    │
    └── reports            <- Generated analysis as PDF, LaTeX, etc.
        └── figures        <- Generated graphics and figures to be used in reporting
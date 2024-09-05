Deep Learning for Network Intrusion Detection System
==============================

In this project we performed experiments on the UNSW-NB15  dataset to create intrusion detection machine learning models for both binary and multiclass classification. Our approach consists of three stages: data preprocessing and feature engineering, baseline machine learning methods for which we selected logistic regression, random forest and k-nearest neighbors, and deep learning methods. The experimental results show that the task of identifying an event of an attack happening could be solved by classical machine learning tools, achieving an accuracy over 99% without a need to employ neural networks. However, the problem of correctly identifying a specific type of attack has proven to be a lot more challenging and the highest balanced accuracy achieved with a deep learning classifier has only reached 77.24%.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for this project.
    ├── data               <- Links and references to the data sources.
    │
    ├── docs               <- Project proposal.
    │   └── papers         <- Annotated papers related to the project.
    ├── models             <- Links and references to the trained models.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0_JQP_EDA`.
    │
    ├── references         <- Explanatory materials.
    │
    └── reports            <- Generated analysis as PDF, LaTeX, etc.
        └── figures        <- Generated graphics and figures to be used in reporting
   


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

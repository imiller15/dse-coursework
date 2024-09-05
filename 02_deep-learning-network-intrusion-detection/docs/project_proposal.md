# PCSC I1910 Project Proposal: Deep Learning for Network Intrusion Detection System
Team members: Temidayo Akinyemi, Pushpen Bikash Goala, Ivan Miller
---

## Motivation
**Clearly state the problem you aim to tackle. Is this an application or a theoretical result? This should be the driving force behind your project.**

Cybersecurity remains an ongoing challenge for businesses, companies, and government institutions, regarding the identification of malicious network traffic and processes running on computers/servers. Attacks like zero-day are one of the most devastating attack types because they do not have any known signature. Today, most cybersecurity analysts look for known attack signatures to detect and respond to cyber threats, but with the large volume of traffic and the speed at which these traffic data and system processes are generated, analysts tend to be overwhelmed and are unable to quickly detect these threats. Typically, this data comes from different security devices like firewalls, privilege management tools, intrusion detection systems, security event and management tools (SIEM). And a lot of time, many of these devices generate false positives which eventually tend to cause alert fatigue on the part of security analysts, ultimately leading to ignoring or missing important alerts. Hence, being able to quickly identify attack patterns requires some machine learning algorithm with predictive capabilities to classify and identify good or bad network traffic/system calls. This will enable quicker response to security incidents and significantly less amount of manual work in detecting the threats.

In this project we will focus on three main areas. First, we will develop a binary classifier to reliably detect an attack/intrusion on the network. From there we’ll be able to focus on a) reducing false positives rates b) go deeper into multiclass classification so that once the attack is detected we could name the type of attack that is happening.


## Methods
**Outline the machine learning techniques you plan to use or improve upon. Clearly state how you plan to use deep learning and how it will differ from the baseline model.**

In this project we intend to work with several machine learning algorithms to develop a performance baseline that we would then try to beat with deep learning methods. During that stage we are planning on applying techniques of cross-validation, hyper-parameter tuning, and potentially ensemble learning to develop the below machine learning models: Logistic Regression, KNN, Random Forest, and SVM. Once we establish the performance baseline we will move on to the main stage of the project and experiment with deep learning models of different architectures from simple sequential models with dense layers to autoencoders and Generative Adversarial Networks (GANs).

## Intended Experiments
**Clearly define the experiments you plan to run and how you will evaluate your machine learning algorithm.**

### Part 1: Data preprocessing, EDA, and feature selection
* Data cleaning - build the dataset, handle missing data and different data types etc.
* EDA and visualizations - familiarize ourselves with the data and 
* Data normalization 

### Part 2: Baseline Methods
* Logistic Regression
* KNN
* Random Forest
* SVM

### Part 3: Deep Learning Methods
* Dense Neural Network - we plan to start with some generic network architecture with few hidden layers, using ReLu activation and since we want to classify as either “normal” or “attack”, then we will use sigmoid activation for the output. We would need our model to at least overfit and then start addressing it. We can increase the network size in case of underfitting and therefore reduce bias.
* Autoencoders
* Generative Adversarial Networks (GANs) 

### Part 4: Model Evaluation
Since most of the data in the UNSW-NB15 represents the negative class with only about 14% of the data points on various types of network attacks we will need to address the imbalance between classes by monitoring precision and recall in addition to the accurate rate and optimizing precision to reduce the chances of false positives.
* Accuracy visualized with a confusion matrix
* Calculate Precision, Recall, and F1-score
* Compute and visualize Receiver Operating Characteristic/Area Under the Curve


### Dataset
We will be using the UNSW-NB15 dataset created by the Cyber Range Lab of UNSW Canberra. 
It was generated using a hybrid of real network packets representing normal network activity and synthetic data on 9 types of attacks: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The  dataset has a total of 2,540,043 records with 49 features including the class label.

### Prior Research
During the work on the project we will review several relevant papers:
* An Analysis of the KDD99 and UNSW-NB15 Datasets for the Intrusion Detection System (Al-Daweri M.S. et al., 2020) https://www.mdpi.com/2073-8994/12/10/1666/pdf
* UNSW-NB15: a comprehensive data set for network intrusion detection systems (Nour M et al., 2015) https://ieeexplore.ieee.org/abstract/document/7348942

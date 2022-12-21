# Capstone - Medical Data

# Project Overview

The field of medical science is one that has evolved over millenia as humans actively seek to understand and treat human ailments and diseases. In this modern, data-driven age, patient data can be a great source of understanding how different internal and external biological features influence the resulting diagnosis of sick patients.

This project aimed to build and compare several binary classification algorithms tasked to classify whether a hospital patient would survive or not, using the patient's medical data. It was not immediately clear what direction this project would take, but rather the outcome and overall problem to solve was determined through investigation and exploratory data analysis.

The project followed the CRISP-DM process, namely:

1. Business (Domain) Understanding
2. Data Understanding
3. Data Exploration and Analysis
4. Data Cleaning and Preparation
5. Modeling
6. Evaluation
7. Deployment

## Project Data

The data used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/sadiaanzum/patient-survival-prediction-dataset?resource=download&select=Dataset.csv). There is no personally identifiable information in this data. The data includes two datasets:

- patient medical data (used for data exploration and machine learning)
- medical data dictionary (used for information and clarity on the patient medical data)

**Please note**: 
I do not own any part of the data used below, and full accreditation to its provision goes to the source publisher on Kaggle (Sadia Anzum). Please use the link provided above to obtain more information about the data used and the author.

**Please note**:
Medical advice and recommendations should **ALWAYS** be solicited from a qualified medical practitioner. This project in no way serves as a medical recommendation or medical advice tool.

## Libraries

The libraries used in this project were libraries commonly associated with data science analysis:

1. Numpy
2. Pandas
3. Matplotlib.pyplot for plotting
5. Various libraries from sklearn (specifically for data imputation, train_test_split, classification algorithms, classification metrics, gridsearch)
4. Seaborn for asthetic purposes
5. Pandas_profiling for the generation of a comprehensive data report (exploration and analysis)
6. Imblearn's SMOTE for classification target label balancing
7. Warnings for notebook neatness
8. Time for logging on model training duration 

**Please note**:
These libraries were imported in the respective sections of the notebook where they were required, and not all at once in the beginning of the notebook.

# Business Understanding: Medical

The nature of medical diagnosis for sick patients is highly contextual, with many potential internal and external factors affecting the overall health and hence medical diagnosis of the patients.

Some factors to consider when diagnosing a patient include:

- demographic
- vitals
- laboratory results (from medical tests)
- comorbidity (multiple diseases in a patient)
- genetic and hereditary influences
- daily lifestyle

A dataset that has information on as many of these factors as possible would go a long way in assisting with the understanding of how each of these factors contributes overall to patient health. Given that this project is concerned with patient survivability, each of these factors **should** be analysed and considered with the probability (or actual) of patient survival in mind.

Something to consider is that many patients that enter hospital premisis requiring care usually do so under traumatic conditions. This results in their inability to provide medical practitioners with all the necessary information that could help ensure their survival. By understanding how a few key factors would affect the probability of patient survival, it is possible to maximise the incoming information for medical practitioners, and hence ensure the best possible care is provided for the patients. This in turn will likely increase the probability that the patients survive.

The benefit of taking a data approach with a problem like this is that statistical and broader data analyses should be able to provide insights as to how each of the aforementioned factors affect each other and the patient as a whole, which is why a comprehensive understanding of the dataset is the next important step in this project.

## APACHE Framework

According to this [site](https://en.wikipedia.org/wiki/APACHE_II#APACHE_III), the APACHE medical framework is a severity-of-disease classification system in the USA, where APACHE stands for "Acute Physiology and Chronic Health Evaluation". APACHE is applied to patients within 24 hours of being admitted into ICU (Intensive Care Unit), and a score is received, where a higher score indicates greater severity of disease, and hence lower probability of patient survival.

The APACHE system comprises of multiple medical measurements and metrics taken from the patient. It has not been validated for use for people under the age of 16. There are multiple levels of APACHE framework, where each successive level is an improvement on the previous iteration. The latest APACHE version is APACHE IV, which was established in 2006. However, earlier versions such as APACHE II are still extensively used due to the availability of comprehensive documentation. Depending on which system is being used, the possible range of the APACHE score changes, where later versions usually have higher number of metrics than previous iterations.

The APACHE framework measurements and scoring system proved to be extremely useful during machine learning model development in this project, comprising of almost the entire group of features used.

# Data Understanding

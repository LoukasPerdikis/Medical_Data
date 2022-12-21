# Capstone - Medical Data

# Project Overview

The field of medical science is one that has evolved over millennia as humans actively seek to understand and treat human ailments and diseases. In this modern, data-driven age, patient data can be a great source of understanding how different internal and external biological features influence the outcome of sick patients.

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

- medical_data.csv (used for data exploration and machine learning)
- medical_data_key.csv (used for information and clarity on the patient medical data)

In order to access these files, please clone this repo and unzip as necessary.

**Please note**: 
I do not own any part of the data used below, and full accreditation to its provision goes to the source publisher on Kaggle (Sadia Anzum). Please use the link provided above to obtain more information about the data used and the author.

Functions were built throughout the project where necessary, and can be found in the respective sections where they are most utilised.

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

Something to consider is that many patients that enter hospital premises requiring care usually do so under traumatic conditions. This results in their inability to provide medical practitioners with all the necessary information that could help ensure their survival. By understanding how a few key factors would affect the probability of patient survival, it is possible to maximise the incoming information for medical practitioners, and hence ensure the best possible care is provided for the patients. This in turn will likely increase the probability that the patients survive.

## APACHE Framework

According to this [site](https://en.wikipedia.org/wiki/APACHE_II#APACHE_III), the APACHE medical framework is a severity-of-disease classification system in the USA, where APACHE stands for "Acute Physiology and Chronic Health Evaluation". APACHE is applied to patients within 24 hours of being admitted into ICU (Intensive Care Unit), and a score is received, where a higher score indicates greater severity of disease, and hence lower probability of patient survival.

The APACHE system comprises of multiple medical measurements and metrics taken from the patient. It has not been validated for use for people under the age of 16. There are multiple levels of APACHE framework, where each successive level is an improvement on the previous iteration. The latest APACHE version is APACHE IV, which was established in 2006. However, earlier versions such as APACHE II are still extensively used due to the availability of comprehensive documentation. Depending on which system is being used, the possible range of the APACHE score changes, where later versions usually have higher number of metrics than previous iterations.

The APACHE framework measurements and scoring system proved to be extremely useful during machine learning model development in this project, comprising of almost the entire group of features used.

The benefit of taking a data approach with a problem like this is that statistical and broader data analyses should be able to provide insights as to how each of the aforementioned factors affect each other and the patient as a whole, which is why a comprehensive understanding of the dataset was the next important step in this project.

# Data Understanding

The most important tool for understanding the data that was used in this project was the medical data key dataset. The data dictionary provides information on how each of the data columns in the patient dataset should be interpreted and what they represent. This was hugely beneficial for someone without formal medical training as it gave an overview of each potential feature column without the need for extensive research, thus saving quite a bit of time in the completion of this project. 

The data was comprised of 186 columns, fragmented into overarching categories. These categories were:

1. Identifier
2. Demographic
3. Labs
4. Vitals
5. Labs Blood Gas
6. APACHE Covariate
7. APACHE Comorbidity
8. APACHE Prediction
9. APACHE Grouping
10. GOSSIS Example Prediction

Most of the categories were not utilised beyond the means of the data exploration and analysis that occurred, this includes all the identifier columns (as there is no need for such identification in a project such as this) as well as the vitals, labs and blood labs gas categories, which showed very high correlation to the APACHE Covariate category. By using the latter category, better metrics which summarised (or indeed, enhanced) the information present in the former categories were used instead.

Ultimately, the most important data in determining patient survivability was certain demographic columns such as patient BMI or whether or not the patient had undergone surgery, alongside the APACHE Covariate columns which contained key medical measurements such as heart rate, blood pressure, gas levels in the blood, and which severe diseases (if any) a patient suffered from during their stay at the hospital.

In terms of potential target variables for the project, three columns were potential candidates:

1. Hospital Death (binary - did the patient die during their hospital stay or not?)
2. APACHE IV Hospital Death Probability (numeric - the APACHE probability that a patient would die during their hospital stay)
3. APACHE IV ICU Death Probability (numeric - the APACHE probability that a patient would die during their ICU stay)

The first of these targets was selected due to ease of use and data cleanliness. No data cleaning was required on the binary column, and there were no missing values. The remaining targets suffered from erroneous values (such as -1 as a probability measure), as well as a small percentage of null values apiece.

The process of understanding the data was rather slow and long, but the extra time spent in the beginning allowed for a highly targeted approach during the exploration and analysis stage which followed. As a result of the data scrutiny, certain 'business' questions were developed and answered.

# Data Exploration and Analysis

## Business Questions from Data Understanding

The following questions were developed as a result of the business and data understanding:

1. What does the probability distribution look like for patient survivability regarding the three potential target variables in the dataset?
2. Is there inherent bias within the dataset? Are there any ethnicities, ages, or genders which are favoured in terms of data representation?
3. What are the most common APACHE comorbidity diseases?
4. Regarding patients with low survival probabilities (APACHE score greater than 50%), what are the mean, min, and max values of their vitals, labs, and APACHE covariates?
5. Which columns suffer from having an excessive amount of nulls? How will these columns be dealt with?

For the in-depth analysis and details on the answers to these questions, please refer to the notebook (Capstone - Medical Data), specifically Section 3.

## Bias

It is an important step in any data science project to acknowledge and deal with potential bias in the data. The notebook has more details on this, but a brief summary will be described here.

- The data was heavily biased towards Caucasian patients (78%), with the next highest represented race being African Americans at 10%.
- The mean age of patients in the dataset was 62 years of age. The distribution (as shown in the notebook) showed a heavy skew towards patients who are older.
- Gender was the demographic measure which suffered from the least amount of bias, where the distribution of male to female data sat at a comfortable 54% / 46% split.

## General Exploration and Analysis

Specific categories in the data were explored and visualised appropriately. Nulls were discovered, alongisde irrelevant or highly correlated columns. This was an important step in the analysis as it helped guide the kind of questions that could be asked and hence, the kind of machine learning problem that could be developed. 

For the in-depth analysis and details on the exploration and analysis, please refer to the notebook (Capstone - Medical Data), specifically Section 3.

## Data Profile

By using the data profile report tool from the [pandas_profiling](https://pandas-profiling.ydata.ai/docs/master/index.html) library, effective correlation, interaction, and general data metrics could be obtained for each of the 186 columns. By this stage of the project, however, some columns had already been removed due to irrelevance (identifier columns) or other reasons.

This is where the only obstacle in the project was encountered: the sheer amount of processing power and memory that the profiling tool requires on a large dataset paced enormous strain on my operating system and hardware, and the generated report (which took the format of an html file) often could not even open in Chrome or Edge. Advanced techniques and setting tweaks were required to open the 757MB html file. Please note that this file is unavailabe in this repo due to size limitations. The code which generates this report can be found in Section 3.2.

The data profile report gave insights which were the deciding factor on which columns could (and ultimately would) be used in machine learning development. Most of the variables were useable, and normally distributed (or at least very close to being normally distributed). There are the odd exceptions where a variable had only a single value (such as readmission_status) and in this case the feature was dropped altogether in order to ensure effective model training.

Some variables such as patient BMI had missing values that could be calculated by using information provided by other variables. As an example, patient BMI can be calculated by using the patient's height and weight information.

Two variables in particular (paco2_apache and paco2_for_ph_apache) were suspiciously similar in terms of distribution and value structure. These two columns were compared further in Section 4 deemed to be identical, therefore one of them was dropped when used in model development. The group of APACHE comorbidty variables were also deemed useful, as they represented severe diseases that would ultimately affect the potential outcome of the patient.

There were numerous variables (particularly in the labs and vitals categories) which contained a high number of missing values (greater than 55%). Even for the variables that were normally (or otherwise) distributed, it would be undesirable to impute such a large number of values. In this instance, it is fortunate that the dataset is comprehensive enough that it provided many metrics of what was essentially the same measurement. For example, the APACHE covariate score variables (classified under the APACHE covariate category) were often highly correlated with measurements of the same name under the labs and vitals categories. What this means is that if the labs or vitals category measurement for heartrate was missing many of its values, it was feasible to use the APACHE score for heartrate instead, as it was still representing the same information. This was a huge benefit to the model development stage of this project, as the correlation coefficients of variables (generated in the profile report) could be referenced to determine if a variable could simply be replaced by a more robust, correlated alternative. For more information on this, please refer to this [article](https://towardsdatascience.com/why-feature-correlation-matters-a-lot-847e8ba439c4).

Features in particular that need to be discussed further are the diagnosis features (apache_3j_bodysystem, apache_2_bodysystem, apache_3j_diagnosis, and apache_3j_diagnosis). These features characterize the diagnosis the patients received upon their admittance into the hospital. While it was a useful piece of information to have, there was a risk that a model might prioritize diagnoses instead of the physical characteristics of each APACHE measurement when assessing a patient's hospital outcome. While certain diagnoses are more severe than others, miracle recoveries do occur and any such bias as a result of diagnosis should be excluded as much as possible. For this reason, the aforementioned columns were not included from any further analysis.

## Null Value Strategy

For variables which could not be replaced via correlation as described above another strategy needed to be implemented in order to deal with missing values. The profile report showed that there were many important variables that were missing a small percentage of their values. When no correlated alternative wass deemed suitable, the distribution of these variables was determined (normal, Bernoulli, etc) and the values imputed accordingly (via MICE - please see further down below). If this was not an option, careful consideration took place as to the variables ultimate importance to the target, and the null values simply dropped if suitable.

## Problem Statement

Through the above analysis and profile report, the problem statement for this project was determined:

How effectively do different machine learning models predict the survival outcome of a patient in the hospital, based on particular metrics that characterize the patient?
The metrics in question include demographic, APACHE covariate, and any labs and vitals not characterized by the APACHE covariate metrics.

In addition to this, the final target and feature dataset configuration was determined. Please refer to the end of Section 3 in the notebook for details on this.

# Data Cleaning and Preparation

Based on the previous section, the necessary steps to preapre and clean the data for machine learning model development took place. First, irrelevant or undesirable columns were removed from the feature list.

Following this, feature engineering took place on the BMI column. In instances where the BMI feature was missing values, calculating the value was attempted by using the patient's height and weight. This is where the second obstacle was encountered: the rows which were missing the BMI value were also missing the height and weight columns. This meant that feature engineering could not take place after all, most unfortunate!

Null values were dealt with next. For binary features, null rows were simply filled with 0s. For other columns, imputation occured via MICE. By using MICE (Multivariate Imputation by Chained Equation), the missing values in the dataset were be filled in by running a series of regression models. These models imputed the missing values sequentially. For more information on how MICE works, please refer to this [article](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87). MICE is particularly useful as it is a non-parametric method of achieving imputation, meaning that values could be filled effectively regardless of the distribution spread that each feature was characterised by. 

This, however is where the third obstacle was encountered: MICE imputation on categorical variables (where the categories defined numerically) , would yield decimal values in place of integer numbers. In order to overcome this, the values in these columns were simply floored using numpy's floor() method.

Finally, the resulting dataset was investigated for duplicate rows. The first instance of each row was kept, and the remaining duplicates removed from the dataset. 

In order to further peruse and investigate the nature of the cleaned dataset for machine learning, the pandas_profile library tool to generate data reports was used once again. For more details, please access the html file in the root folder (final_medical_data_profile.html).

# Modeling

## SMOTE

First, the target and feature columns were separated and sklearn's train_test_split class was used to generate the appropriate training and testing datasets. Upon inspection of the training dataset targets, it was found that a huge disparity existed between the number of the one class label when compared to the other. In other words, any model training on such a dataset would know how to classify one label far better than the other as there was not an even distribution of target labels.

This was the fourth obstacle, and was dealt with elegantly using SMOTE. SMOTE aims to balance the target labels to include more patient deaths than currently exist, ensuring a well balanced model. This is achieved by upsampling the target label with fewer counts by generating data that is statistically similar to other rows of the same target. By the end of the SMOTE application, there was an even distribution of target labels that the models could learn from.

## Machine Learning

As declared by the problem statement, the purpose of this project is to compare several binary classification models in their ability to predict whether or not a hospital patient would die. To this end, the classification algorithms used were sklearn's:

1. Gaussian Naive Bayes
2. Logistic Regression
3. K-Nearest Neighbors 
4. Random Forest

Multiple parameters for each classifier was also established, for the purpose of Gridsearch training. Please note, however, that parameter selection options were determined with requirements of efficiency and simplicity. This project is concerned more with the comparison of different models than the optimization of a single model, therefore parameter variation was slim and not of the highest priority.

The trained models were saved in a dictionary, were orderly model evaluation could take place.

# Evaluation

## Metrics Discussion

Suitable classification algorithm metrics included model accuracy, precision, recall and F1 scores. A confusion matrix heatmap was also be of great assistance in determining how well the models performed in terms of Type I and Type II errrors. In addition to this, the ROC curve alongisde the AUC score was be used to determine how well the model has performed in classifying unseen data.

In general, the closer the scores were to 1, the better the model had performed. Conversely, the closer the scores were to 0, the worse the model had performed. This applied to the accuracy, precision, f1, recall, and AUC scores.
To interpret the ROC curve, the closer the curve appeared to the top left of the plot (i.e., the larger the area under the curve), the better the model had performed.

The reason why these metrics were chosen was due to what they represent relative to classification machine learning problems, as outlined in this [article](https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/). Before the metrics can be discussed below, a short description of the difference between false and true positives and negatives is provided:

**True Positive**: Extent to which model correctly predicts the positive class.

**True Negative**: Extent to which the model correctly predicts the negative class.

**False Positive**: Extent to which the model incorrectly predicts a negative 
class as positive.

**False Negative**: Extent to which the model incorrectly predicts a positive 
class as negative.

## Accuracy

Accuracy represents how well a model is able to achieve an exact match when comparing its predicted values and the true values. It is calculated through the ratio of true positives and true negaties to all positive and negative data points. The score produced gives an idea of how accurately the model can predict the **correct** outcome, given the appropriate datapoints. Accuracy does not give any information regarding the nature of the produced model's errors.

## Precision

This is the proportion of positively predicted labels that are indeed correct. Unfortunately, precision is affected by class distribution, which means that it will be lower if there are fewer of one kind of class than the other. While the training dataset underwent SMOTE in order to balance class labels, no such exercise was performed on the testing dataset (or any other biased prediction set that may exist). For this reason however, precision can be a very useful indicator when classes are very imbalanced. It is calculated by taking the ratio of the true positives to the sum of false positives and true positives. It is a great metric to see how well the model can avoid false positives.

## Recall

This metric represents the model's ability to predict the positive values out of all actual positive values. A high recall shows that a model can easily identify true positive data examples, and opposite is true if the recall score is low. It is calculated by taking the ratio of true positives to the sum of false negatives and true positives. Recall is slightly different to precision, in that precision is concerned with the proportion of positively predicte labels, while the recall actually needs to correctly identify what those positive labels are. It is a great metric to see how well the model can avoid false negatives.

## F1

This metric is a function of the recall and precision metrics. It is akin to accuracy in that it gives a single value which can be interpreted as the model's overall quality at a high level. It is calculated through the following formula: 
F1 = 2 * Precision * Recall / (Precision + Recall)

It is an extremely useful metric in cases where models are optimized for recall or precision, but such a use case is beyond the scope of this project.

## Model Performance Discussion

In terms of **accuracy**, the RF model was clearly the best with a score of 91%. All other models scored in the high 70s, with the worst performaing model (in terms of accuracy) being KNN at 74%. For instances where accuracy is the only metric that matters, RF is the obvious choice.

In terms of **precision**, once again the RF model is superior with a score of 52%. It is alarming that the highest precision score is only 52%, however, as this shows that half the time the model predicts a patient would die, they actually would not die. The other models perform even worse, with values ranging from 16% (KNN once again) to 23%. This means that for these models, patients would survive between 84% and 77% of the time the model has determined they would not.

In terms of **recall**, the LR model scored the highest with 70%. It is worth noting that for this metric, the previously superior RF model scored dismally low, only managing to correctly predict positive values 28% of the time. It is worth noting that the other models also scored higher than RF in recall, at 49% for KNN and 66% for NB. This shows that while RF is relatively good at avoiding false positives (compared to the other trained models), it is hopelessly outshined when predicting false negatives. This kind of situation eloquently demonstrates why the precision-recall tradeoff is important, and why multiple metrics need to be taken into account when evaluating model performance.

In terms of **F1**, RF once again scores the highest at 36% (which is not good in any case). The other models are not too far behind (ranging between 24% and 35%). The overall low F1 scores are to be expected, as the recall and precision scores for all the models were not simultaneously high. Therefore, despite the high accuracy of the models, all are severely lacking in their ability to discern between false positives and false negatives.

In terms of **ROC-AUC**, RF scores the highest once again with a value of 85%. The lowest scoring model for this metric is the KNN model 67%. These scores mean that each model does an acceptable job of classifiying the data, with RF scoring it better than the others. It should also be noted that the LR model also received a high ROC-AUC score, at 82%.

## Conclusion

Despite poor performance across the board in terms of false postive and flase negative discernment, all of the models did an acceptable job of classifying unseen data. The poor recall and precision scores could be improved with further analysis and model tweaking, however this is beyond the scope of this project.

Overall, the Random Forest classifier was the highest performing model, having obtained the best accuracy, precision and ROC-AUC scores, but performing the worst in recall. The produced RF model's accuracy and ROC-AUC are deemed sufficiently high for the purposes of the problem statement declared in Section 3.

# Deployment

A technical article has been written and published on Medium. It can be found [here]()INSERT ARTICLE 

**Please note**:
Medical advice and recommendations should **ALWAYS** be solicited from a qualified medical practitioner. This project in no way serves as a medical recommendation or medical advice tool.


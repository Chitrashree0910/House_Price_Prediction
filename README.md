# Overview
This project aims to predict house prices based on various features such as location, size, number of rooms, and other attributes. Using Linear Regression, we build a predictive model that estimates house prices based on historical data. The project follows a structured Machine Learning workflow, covering data preprocessing, feature engineering, model training, and evaluation.

## Tools & Technologies Used
For this <b>House Price Prediction</b> project, I leveraged several key tools and technologies to explore, analyze, and model the data effectively:
- <b>Python</b>: The core programming language used for data analysis, visualization, and machine learning
    - <b>Pandas</b>: Used for data manipulation, handling missing values, and performing exploratory data analysis (EDA)
    - <b>NumPy</b>: Assisted in numerical computations and handling arrays efficiently
    - <b>Matplotlib</b>: Created visual representations of the data to identify patterns and trends
    - <b>Seaborn</b>: Enhanced data visualization with more advanced plots and statistical insights
- <b>Jupyter Notebook</b>: Served as the interactive environment for coding, visualization, and documentation
- <b>Visual Studio Code (VS Code)</b>: Used for writing, editing, and debugging Python scripts
- <b>Git & GitHub</b>: Essential for version control, tracking changes, and sharing the project publicly

## Project Steps
<b>Step 1</b>: Exploratory Data Analysis (EDA)

<b>Step 2</b>: Feature Selection

<b>Step 3</b>: Feature Engineering

<b>Step 4</b>: Model Selection & Training

<b>Step 5</b>: Model Evaluation

<b>Step 6</b>: Model Optimization

<b>Step 7</b>: Deployment

## Step 1: Exploratory Data Analysis (EDA)
### Data Analysis
- Checked missing values and filled them using:
    - <b>Numerical columns</b>: Filled with the median
    - <b>Categorical columns</b>: Filled with the mode
- Identified and removed duplicate rows
- Examined distributions of numerical variables

View my notebook with detailed steps here:
[1_Exploratory_Data_Analysis.ipynb](Project_Files\1_Exploratory_Data_Analysis.ipynb)

### Data Visualization
<b>Distribution of House Prices (Before & After Cleaning)</b>
```python
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # Before cleaning
    sns.histplot(original_df['price'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Before Data Cleaning')

    # After cleaning
    sns.histplot(cleaned_df['price'], bins=30, kde=True, ax=ax[1])
    ax[1].set_title('After Data Cleaning')

    plt.show()
```

### <b>Result</b>

![Before & After Data Cleaning](Images\1_EDA_Before_&_After_Data_Cleaning.png) 
*Before & After Data Cleaning*

### Insights
- House prices showed <b>strong correlation</b> with variables like square footage and number of bedrooms.

- Outliers and missing values <b>significantly affected data distribution</b>, requiring careful handling.

- Cleaning the data resulted in a <b>more stable and reliable dataset</b> for modeling.
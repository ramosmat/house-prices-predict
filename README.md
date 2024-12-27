# House Price Prediction

This repository contains a machine learning project that predicts house prices based on various features. Using a comprehensive dataset and regression models, the project demonstrates how to preprocess data, analyze relationships, and build predictive models for continuous target variables.

## Dataset

The dataset used in this project includes features describing houses and their surroundings, such as:

- **Lot Area**: Size of the lot in square feet.
- **Year Built**: The year the house was built.
- **Overall Quality**: Rates the overall material and finish of the house.
- **Total Rooms**: Total number of rooms excluding bathrooms.
- **Neighborhood**: Physical location within the city.
- **SalePrice**: The target variable representing the sale price of the house.

The dataset can be found at [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Key Libraries

The following Python libraries were used in this project:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For implementing regression models and preprocessing steps.

## Machine Learning Models

Regression models were implemented and evaluated:

- **TensorFlow Decision Forests (TFDF)**

## Project Workflow

1. **Data Preprocessing**:
   - Handling missing values through imputation for both numerical and categorical features.

2. **Exploratory Data Analysis (EDA)**:
   - Verifying how house prices are distributed

3. **Model Training and Evaluation**:
   - Splitting the data into training and testing sets.
   - Training regression models.
   - Evaluating models using metrics such as OOB Score (RMSE) and MSE - Mean Squere Error.

4. **Results**:
   - RMSE reached a score of **29309.21** while MSE reached **637292672.00**.

## Results Summary

Key insights derived from the project:
- **Tensor Flow Decision Forests** are among the most significant predictors of house prices.
- Newer houses and those located in high-value neighborhoods tend to have higher sale prices.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/ramosmat/house-prices-predict.git
   ```

2. Navigate to the project directory:
   ```bash
   cd house-prices-predict
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook script.ipynb
   ```

## Contributing

Contributions are welcome! If you have ideas to improve this project, feel free to submit issues or pull requests.

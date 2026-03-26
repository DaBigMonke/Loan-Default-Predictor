# Loan Default Predictor

A Python ML pipeline that cleans credit risk data, trains a Decision Tree classifier to predict loan defaults, and serves predictions through a custom circular doubly-linked list interface.

---

## Overview

This project takes a real-world credit risk dataset and builds an end-to-end pipeline: from raw data cleaning all the way to an interactive prediction interface. Given a set of loan applicants, the model predicts whether each one is likely to default — and recommends Accept or Reject accordingly.

## Features

- Cleans raw CSV data by removing incomplete records and filtering outliers (e.g. applicants with unrealistic ages)
- Scales income and loan amount features using `StandardScaler`
- Trains a `DecisionTreeClassifier` on three features: loan amount, income, and credit history length
- Evaluates model performance with accuracy score, precision/recall report, and confusion matrix
- Generates a histogram of loan defaults by age group and a pie chart of default rates among homeowners
- Serves predictions through an interactive CLI interface backed by a custom **circular doubly-linked list**, allowing you to scroll through applicants one by one

## Tech Stack

- Python
- scikit-learn
- Matplotlib

## Project Structure

```
├── main.py                 # Main pipeline: cleaning, training, evaluation, deployment
├── carousel.py             # Circular doubly-linked list implementation
├── credit_risk_train.csv   # Training data
├── credit_risk_test.csv    # Test data
├── loan_requests.csv       # New applicants to predict on
```

## How to Run

1. Clone the repo and navigate to the project folder
2. Install dependencies:
   ```bash
   pip install scikit-learn matplotlib
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. The program will clean the data, train the model, print evaluation metrics, display charts, and launch the interactive predictor

## Sample Output

```
Initial number of rows: 28638
Remaining number of rows: 28205

Model Evaluation on Scaled Test Set:
Accuracy: 0.XX

Predicted Loan Status for Requests:
[0, 1, 0, 1, ...]

--------------------------------------------------
Borrower: Jane Doe
Age: 34
Income: $52000
Loan Amount: $8000
Predicted loan_status: Will not default
Recommend: Accept
--------------------------------------------------
```

## Data Structure

The `Carousel` class in `carousel.py` is a circular doubly-linked list. Each node stores one loan applicant's data along with their predicted status. The interface lets you navigate forward and backward through all applicants in a loop — wrapping around when you reach either end.

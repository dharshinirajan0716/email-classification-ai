# AI-Based Email Classification System

An end-to-end NLP + Machine Learning project that classifies email subject/body text into departments/categories and provides a Streamlit dashboard.

## Important note about the dataset
`data/sample_emails.csv` is a **small synthetic demo dataset** so the project can run immediately. It is not a real Kaggle dataset and its evaluation scores are not suitable for reporting as real-world performance. For your final submission, download a public email classification dataset from Kaggle and replace/extend this data.

## Requirements
- Python 3.10 or newer recommended
- Internet only needed once to install Python packages (and to download your chosen Kaggle dataset)

## Setup on Windows
1. Extract the ZIP file.
2. Open the extracted `Email_Classification_Project` folder.
3. In File Explorer, click the address bar, type `cmd`, and press Enter.
4. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If `python` does not work, try `py` instead of `python`.

## Dataset format
The included demo CSV has columns:
- `subject`: email subject
- `body`: email body
- `category`: target label

If your downloaded dataset uses different column names, edit `DATA_PATH`, `SUBJECT_COLUMN`, `BODY_COLUMN`, and `LABEL_COLUMN` at the top of `train_model.py` and `app.py`, or rename the columns in your CSV to `subject`, `body`, and `category`.

## What the project does
1. Inspects the dataset and reports missing values, duplicates, sample count, and category distribution.
2. Cleans subject/body text, handles missing values, removes duplicate records.
3. Uses TF-IDF text features and compares Multinomial Naive Bayes, Logistic Regression, and Random Forest.
4. Saves the best model pipeline and metrics to `models/` and `reports/`.
5. Offers a Streamlit dashboard for predictions, confidence scores (when supported), and session prediction history.

## Outputs
- `models/email_classifier.joblib`: selected trained model pipeline
- `reports/dataset_analysis.txt`: dataset analysis
- `reports/model_comparison.csv`: model metrics
- `reports/classification_report.txt`: detailed report for the selected model
- `reports/confusion_matrix.png`: confusion matrix image

## Limitations
- A confidence score is a model score, not a guarantee that the prediction is correct.
- With a small dataset, accuracy can vary substantially. Use a larger real dataset and test it carefully.
- The app predicts categories only; it does not send or route real emails automatically.

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
MODEL_PATH = 'Artifacts/model_data.joblib'

model_data=joblib.load(MODEL_PATH)
model=model_data['model']
scaler=model_data['scaler']
cols_to_scale=model_data['cols_to_scale']
features=model_data['features']



def prepare_data(age,income,loan_amount,loan_to_income,loan_tenure_months,avg_dpd_per_delinquency
    ,delinquency_ratio,credit_utilization_ratio,number_of_open_accounts,residence_type,loan_purpose,loan_type):
    input_data={
        'age':[age],
        'income':[income],
        'loan_amount':[loan_amount],
        'loan_to_income':[loan_to_income],
        'loan_tenure_months':[loan_tenure_months],
        'avg_dpd_per_delinquency':[avg_dpd_per_delinquency],
        'delinquency_ratio':[delinquency_ratio],
        'credit_utilization_ratio':[credit_utilization_ratio],
        'number_of_open_accounts':[number_of_open_accounts],
        'residence_type_Owned':[1 if residence_type=='Owned' else 0],
        'residence_type_Rented':[1 if residence_type=='Rented' else 0],
        'loan_purpose_Education':[1 if loan_purpose=='Education' else 0],
        'loan_purpose_Home': [1 if loan_purpose=='Home' else 0],
        'loan_purpose_Personal': [1 if loan_purpose=='Personal' else 0],
        'loan_type_Unsecured':[1 if loan_type=='Unsecured' else 0],
        'number_of_closed_accounts':[1],#Duumy values for scaling
        'enquiry_count':[1],#Duumy values for scaling
        'number_of_dependants':[1],#Duumy values for scaling
        'years_at_current_address':[1],#Duumy values for scaling
        'sanction_amount':[1],#Duumy values for scaling
        'processing_fee':[1],#Duumy values for scaling
        'gst':[1],#Duumy values for scaling
        'net_disbursement':[1],#Duumy values for scaling
        'principal_outstanding':[1],#Duumy values for scaling
        'bank_balance_at_application':[1]#Duumy values for scaling
        }

    df=pd.DataFrame(input_data)#Creating DataFrame
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])
    df=df[features]
    return df

def predict(age,income,loan_amount,loan_to_income,loan_tenure_months,avg_dpd_per_delinquency
    ,delinquency_ratio,credit_utilization_ratio,number_of_open_accounts,residence_type,loan_purpose,loan_type):

    input_df=prepare_data(age,income,loan_amount,loan_to_income,loan_tenure_months,avg_dpd_per_delinquency
    ,delinquency_ratio,credit_utilization_ratio,number_of_open_accounts,residence_type,loan_purpose,loan_type)

    probability, credit_score, rating =calculate_credit_score(input_df)

    return probability, credit_score, rating

def calculate_credit_score(input_df,base_score=300,scale_length=600):

    default_probability=model.predict_proba(input_df)[:,1]

    non_default_probability=1-default_probability

    credit_score=base_score+(non_default_probability*scale_length)

    def get_rating(score):
        if 300<=score<=500:
            return 'poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating=get_rating(credit_score)

    return default_probability,credit_score,rating







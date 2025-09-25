#Full year  Business forecast 2025
#Import Essential Libraries
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

st.title("Full Year Business Forecast (2025)")
#Load the Data
uploaded_file=st.file_uploader("Upload MIS Data Excel File",type=["xlsx"])
if uploaded_file:
    df=pd.read_excel(uploaded_file,sheet_name="Sheet1")
    df['Reg.Date']=pd.to_datetime(df['Reg.Date'],errors='coerce')
    df['YearMonth']=df['Reg.Date'].dt.to_period('M')
    #st.dataframe(df)
    #Filter Options
    technologies=df['Subject'].dropna().unique()
    colleges=df['College'].dropna().unique()
    locations=df['Location'].dropna().unique()
    selected_tech=st.selectbox("Select Technology",sorted(technologies))
    selected_college=st.selectbox("Select College (Optional)",['All']+sorted(colleges.tolist()))
    selected_location=st.selectbox("Select Location (Optional)",['All']+sorted(locations.tolist()))
    #Apply Filters
    data=df[df['Subject']==selected_tech]
    if selected_college!='All':
        data=data[data['College']==selected_college]
    if selected_location!='All':
        data=data[data['Location']==selected_location]
    #st.dataframe(data)
    
    #grouping by month
    monthly=data.groupby('YearMonth').size().reset_index(name='SNo.')
    monthly=monthly.set_index('YearMonth').asfreq('M').fillna(0)
    monthly.index=monthly.index.to_timestamp()
    #st.dataframe(monthly)
    if len(monthly)>=2:
        #Prepare Regression Model
        X=np.array([d.toordinal() for d in monthly.index]).reshape(-1,1)
        #st.dataframe(X)
        y=monthly['SNo.'].values
        model=LinearRegression()
        model.fit(X,y)
        #Predict All Months of 2025
        future_date=pd.date_range(start="2025-01-01",end="2025-12-01",freq='MS')
        X_future=np.array([d.toordinal() for d in future_date]).reshape(-1,1)
        y_pred=model.predict(X_future)
        #st.dataframe(y_pred)
        
        
        #Showing Predicted values
        forecast_df=pd.DataFrame({"month":future_date.strftime('%B %Y'),"Predicted_Enrollments":np.round(y_pred).astype(int)})
        st.subheader("Monthly Prediction For 2025")
        st.dataframe(forecast_df)
        #show total prediction
        st.success(f"Total Predicted Enrollments for 2025 :{int(np.round(y_pred).sum())}")
    else:
        st.warning("Not Enough Data For Prediction")
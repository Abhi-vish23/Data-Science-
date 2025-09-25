#Language VS Placement Correlation
import streamlit as st 
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

st.title("Language VS Placement Correlation")
uploaded_file=st.file_uploader("Upload File",type=['csv'])

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    #Data Preprocessing
    df["Known_Languages"]=df["Known_Languages"].apply(lambda x:x.split(', '))
    mlb=MultiLabelBinarizer()
    language_df=pd.DataFrame(mlb.fit_transform(df['Known_Languages']),columns=mlb.classes_)
    df=pd.concat([df,language_df],axis=1)
    #Model
    X=df[language_df.columns]
    y=df['Package_LPA']
    model=DecisionTreeRegressor()
    model.fit(X,y)
    #prediction
    st.subheader("Predict Salary based on Known Language")
    selected_langs=st.multiselect("Select Language you know: ",options=X.columns)
    button=st.button("Predict")
    if button:
        #st.markdown("OK")
        input_data=[1 if lang in selected_langs else 0 for lang in X.columns]
        predicted_package=model.predict([input_data])[0]
        st.success(f"Predicted Package : {predicted_package:.2f} LPA")
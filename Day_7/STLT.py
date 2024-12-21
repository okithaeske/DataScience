import streamlit as st
import pandas as pd
import plotly.express as px

st.title("This is a Title")
st.header("This is a Header")
st.subheader("This is a subheader")
st.text("Streamlit text")

data = {
    "Product": ["Laptop", "Smartphone", "Headphones", "Keyboard", "Monitor"],
    "Category": ["Electronics", "Electronics", "Accessories", "Accessories", "Electronics"],
    "Price": [1200, 800, 50, 30, 200],
    "Stock": [50, 100, 200, 150, 80],
    "Sales": [30, 70, 150, 100, 60]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Select Box option
st.subheader("select BOX")
selected_catergory=st.selectbox("Select the product category",df["Category"].unique())
catergory_df=df[df["Category"]==selected_catergory]
st.dataframe(catergory_df) 

# Slider
st.subheader("Slider")
price_Filter=st.slider("Filter from prices",0,1200,(0,800))
Filtered_price=df[(df["Price"]>=price_Filter[0])&(df["Price"]<price_Filter[1])]
st.dataframe(Filtered_price)

st.subheader("Buttons")

if st.button("Show all products"):
    st.write("text")
    st.dataframe(df)
else:
    st.write("click on the button")

st.subheader("Date")
selected_Date = st.date_input("select ")


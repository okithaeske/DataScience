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

st.subheader("Radio buttons")
selection = st.radio("Select one from here",  ["Sales", "Stock", "Price"])
st.write(f"You have selected {selection}")
r_df = df[selection]
st.dataframe(r_df)


st.subheader("Checkboxes")
if st.checkbox("Show sales data"):
    st.bar_chart(df.set_index("Product")["Sales"])


st.subheader("Bar Graph")
bar = px.bar(df, x="Product", y=["Stock"], title="Stocks Graph")
st.plotly_chart(bar)

st.subheader("Bar Graph")
line = px.line(df, x="Product", y="Price", title="Price Graph")
st.plotly_chart(line)

#show items using a slider
filter1 = st.slider("Filter from Sales: ", df["Sales"].min(), df["Sales"].max(), (0,(333)))
f_df = df[(df["Sales"] >= filter1[0]) & (df["Sales"] <= filter1[1])]
bar2 = px.bar(f_df, x="Product", y=["Sales"], title="Sales Graph")
st.plotly_chart(bar2)

st.subheader("Pie Chart")
pie = px.pie(df, names="Category", values="Sales", title="Sales Chart")
st.plotly_chart(pie)



st.sidebar.selectbox("Choose an option",["Option 1","Option 2","Option 3","Option 4"])

uploaded_file = st.file_uploader("Choose a file",type=["csv","xlsx"])
# Error handling
if uploaded_file is not None:
    try:
        # Try reading the uploaded file
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        
        # Display the dataframe
        st.write(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a file.")
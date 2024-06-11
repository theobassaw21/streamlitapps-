import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)
# Function to load data
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Sidebar for file upload
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Sidebar filters
    st.sidebar.header('Filter Options')

    # Date column selection
    date_column = st.sidebar.selectbox('Select date column', options=data.columns)

    # Convert selected date column to datetime
    data[date_column] = pd.to_datetime(data[date_column])

    # Date range filter
    start_date = st.sidebar.date_input('Start date', data[date_column].min())
    end_date = st.sidebar.date_input('End date', data[date_column].max())

    # Weather parameter columns selection
    #weather_columns = st.sidebar.multiselect('Select weather parameter columns', options=data.columns)

    # Plot type selection
    plot_type = st.sidebar.selectbox('Select plot type', ['Line Plot', 'Bar Plot', 'Scatter Plot', 'Box Plot','Polar Plot'])

    # X and Y columns selection for plotting
    x_column = st.sidebar.selectbox('Select x-axis column', options=data.columns)
    y_column = st.sidebar.selectbox('Select y-axis column', options=data.columns)

    # Filter data based on user input
    filtered_data = data[(data[date_column] >= pd.to_datetime(start_date)) & (data[date_column] <= pd.to_datetime(end_date))]

    # Main Panel
    st.title('Weather Analysis Dashboard')
    st.write(f"Analyzing data from {start_date} to {end_date}")

    # Plotting
    st.subheader(f"{plot_type} of {y_column} vs {x_column}")
    if plot_type == 'Line Plot':
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=filtered_data, x=x_column, y=y_column)
        plt.title(f"{y_column} vs {x_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot(plt)
    elif plot_type == 'Bar Plot':
        fig = px.bar(filtered_data, x=x_column, y=y_column)
        st.plotly_chart(fig)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(filtered_data, x=x_column, y=y_column)
        st.plotly_chart(fig)
    elif plot_type == 'Box Plot':
        fig = px.box(filtered_data, x=x_column, y=y_column)
        st.plotly_chart(fig)
    elif plot_type == 'Polar Plot':
        fig = px.bar_polar(filtered_data, r=y_column, theta=x_column, title=f"{y_column} vs {x_column}", color_discrete_sequence=px.colors.sequential.Plasma_r)
        st.plotly_chart(fig)
    

    # Display raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(filtered_data)
        #st.subheader('DATA SIZE')
        #st.write(data.size)
    
    # PyGWalker integration
   # st.subheader('Interactive Data Exploration with PyGWalker')
    #pyg.walk(data)

 # Check if 'rain' or 'precipitation' parameter exists in the dataset
   
 










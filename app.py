import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid

# @st.cache_data
# def load_data():
#     return pd.read_csv('Clicked_On_AD.csv')

# df = load_data()

# st.title('Exploratory Data Analysis (EDA) App')

# # Display basic info
# st.subheader('Dataset Overview')
# st.write(f'Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')
# st.write('Columns:', df.columns)

# # Display sample data
# st.subheader('Sample Data')
# st.write(df.head())

# # Descriptive statistics
# st.subheader('Descriptive Statistics')
# st.write(df.describe())

# # Histogram
# st.subheader('Histogram for a Numerical Column')
# column = st.selectbox('Select Column', df.select_dtypes(include=['number']).columns)
# fig, ax = plt.subplots()
# ax.hist(df[column].dropna(), bins=30, color='skyblue', edgecolor='black')
# ax.set_title(f'{column} Distribution')
# st.pyplot(fig)

# # Correlation heatmap
# st.subheader('Correlation Heatmap')
# corr_matrix = df.select_dtypes(include=['number']).corr()
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
# st.pyplot(fig)

# # Display interactive data table
# st.subheader('Interactive Data Table')
# AgGrid(df)


# import streamlit as st
# import pandas as pd
# import pygwalker as pyg

# # Load dataset function
# @st.cache_data
# def load_data():
#     return pd.read_csv('Clicked_On_AD.csv')  # Replace with your dataset's path

# # Load data into pandas dataframe
# df = load_data()

# # Display Streamlit title and header
# st.title('Exploratory Data Analysis with PyGWalker')

# # Basic info about the dataset
# st.subheader('Dataset Overview')
# st.write(f'Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')
# st.write(f'Columns: {df.columns}')

# # Sample of the data
# st.subheader('Sample Data')
# st.write(df.head())

# # Display PyGWalker interactive dashboard
# vis_select = st.sidebar.checkbox("**Is visualisation required for this dataset?**")

# if vis_select:

#     st.write( '### 3. Visual Insights ')

#     #Creating a PyGWalker Dashboard
#     walker = pyg.walk(df)
#     # st.components.v1.html(walker, width=1100, height=800)  #Adjust width and height as needed
#Importing the necessary packages


import streamlit as st
import openpyxl
import pygwalker as pyg
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

#Setting up web app page
st.set_page_config(page_title='Exploratory Data Analysis App', page_icon=None, layout="wide")

#Creating section in sidebar
st.sidebar.write("****A) File upload****")

#User prompt to select file type
ft = st.sidebar.selectbox("*What is the file type?*",["Excel", "csv"])

#Creating dynamic file upload option in sidebar
uploaded_file = st.sidebar.file_uploader("*Upload file here*")

if uploaded_file is not None:
    file_path = uploaded_file

    if ft == 'Excel':
        try:
            #User prompt to select sheet name in uploaded Excel
            sh = st.sidebar.selectbox("*Which sheet name in the file should be read?*",pd.ExcelFile(file_path).sheet_names)
            #User prompt to define row with column names if they aren't in the header row in the uploaded Excel
            h = st.sidebar.number_input("*Which row contains the column names?*",0,100)
        except:
            st.info("File is not recognised as an Excel file")
            sys.exit()
    
    elif ft == 'csv':
        try:
            #No need for sh and h for csv, set them to None
            sh = None
            h = None
        except:
            st.info("File is not recognised as a csv file.")
            sys.exit()

    #Caching function to load data
    @st.cache_data()
    def load_data(file_path,ft,sh,h):
        
        if ft == 'Excel':
            try:
                #Reading the excel file
                data = pd.read_excel(file_path,header=h,sheet_name=sh,engine='openpyxl')
            except:
                st.info("File is not recognised as an Excel file.")
                sys.exit()
    
        elif ft == 'csv':
            try:
                #Reading the csv file
                data = pd.read_csv(file_path)
            except:
                st.info("File is not recognised as a csv file.")
                sys.exit()
        
        return data

    data = load_data(file_path,ft,sh,h)

#=====================================================================================================
## 1. Overview of the data
    st.write( '### 1. Dataset Preview ')

    try:
      #View the dataframe in streamlit
      st.dataframe(data, use_container_width=True)

    except:
      st.info("The file wasn't read properly. Please ensure that the input parameters are correctly defined.")
      sys.exit()

## 2. Understanding the data
    st.write( '### 2. High-Level Overview ')

    #Creating radio button and sidebar simulataneously
    selected = st.sidebar.radio( "**B) What would you like to know about the data?**", 
                                ["Data Dimensions",
                                 "Field Descriptions",
                                "Summary Statistics", 
                                "Value Counts of Fields"])

    #Showing field types
    if selected == 'Field Descriptions':
        fd = data.dtypes.reset_index().rename(columns={'index':'Field Name',0:'Field Type'}).sort_values(by='Field Type',ascending=False).reset_index(drop=True)
        st.dataframe(fd, use_container_width=True)

    #Showing summary statistics
    elif selected == 'Summary Statistics':
        ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
        st.dataframe(ss, use_container_width=True)

    #Showing value counts of object fields
    elif selected == 'Value Counts of Fields':
        # creating radio button and sidebar simulataneously if this main selection is made
        sub_selected = st.sidebar.radio( "*Which field should be investigated?*",data.select_dtypes('object').columns)
        vc = data[sub_selected].value_counts().reset_index().rename(columns={'count':'Count'}).reset_index(drop=True)
        st.dataframe(vc, use_container_width=True)

    #Showing the shape of the dataframe
    else:
        st.write('###### The data has the dimensions :',data.shape)

    import streamlit as st
    import altair as alt
    import pandas as pd

    # Sample DataFrame
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 25, 30, 40]
    })

    # Create an Altair plot
    chart = alt.Chart(data).mark_line().encode(
        x='x',
        y='y'
    ).properties(title="Altair Line Plot")

    # Display the Altair plot in Streamlit
    st.altair_chart(chart, use_container_width=True)

    import streamlit as st
    import plotly.express as px

    # Load sample data
    data = px.data.iris()

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(data, x='sepal_width', y='sepal_length', color='species', title="Plotly Iris Data")

    # Display the Plotly plot in Streamlit
    st.plotly_chart(fig)


#=====================================================================================================
## 3. Visualisation

    #Selecting whether visualisation is required
    vis_select = st.sidebar.checkbox("**C) Is visualisation required for this dataset?**")

    if vis_select:

        st.title("Use Pygwalker In Streamlit")
 
        # You should cache your pygwalker renderer, if you don't want your memory to explode
        @st.cache_resource
        def get_pyg_renderer() -> "StreamlitRenderer":
            
            # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
            return StreamlitRenderer(data, spec="./gw_config.json", spec_io_mode="rw")
 
        
        renderer = get_pyg_renderer()
        
        renderer.explorer()

        # st.write( '### 3. Visual Insights ')

        # #Creating a PyGWalker Dashboard
        # pyg.walk(data)
        
        # # st.components.v1.html(walker, width=1100, height=800)  #Adjust width and height as needed
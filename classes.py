# importing libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class DataLoader:
    """Class to Load data"""
    def __init__(self):      
        pass

    def read_data(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        """
        Load data from a CSV, TSV, JSON or Excel file
        """
        if file_ext == '.csv':
            return pd.read_csv(file_path, index_col=None)
        
        elif file_ext == '.tsv':
            return pd.read_csv(file_path, sep='\t')

        elif file_ext == '.json':
            return pd.read_json(file_path)

        elif file_ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)

        else:
            raise ValueError(f"Unsupported file format:")
       
class DataInfo:
    """Class to get dataset information """
    def __init__(self):      
        pass

    def info(self, df): 
        """
        Displaying Relevant Information on the the Dataset Provided
        """    
        # Counting no of rows 
        print('=='*20 + f'\nShape of the dataset : {df.shape} \n' + '=='*20 + '\n')
        
        # Extracting column names
        column_name =  df.columns 
        print('=='*20 + f'\nColumn Names\n' + '=='*20 +  f'\n{column_name} \n ')

        # Checking if 'Timestamp' column exists and displaying date range
        if 'Timestamp' in df.columns:
            print("=="*20 + "\nRange of the Dataset (Timestamp)\n" + "=="*20)
            # Converting Timestamp column to datetime if not already
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            print("Start Date:", df['Timestamp'].min())
            print("End Date:  ", df['Timestamp'].max())
        print('\n')

        # List of all numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        print('=='*20 + f'\nNumerical Columns\n' + '=='*20)
        print(numerical_cols, end="\n\n")
        
        # List of all categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        print('=='*20 + f'\nCategorical Columns\n' + '=='*20)
        print(categorical_cols, end="\n\n")
            
        # Data type info
        print('=='*20 + f'\nData Summary\n' + '=='*20 )
        data_summary = df.info() 
        print('=='*20 +'\n')

        # Descriptive statistics
        describe =  df.describe() 
        print('=='*20 + f'\nDescriptive Statistics\n' + '=='*20  )
        display(describe)
        
        #Display the dataset
        print('=='*20 + f'\nDataset Overview\n'+ '=='*20 )
        return df.head()
    
class DataChecks:
    """Class to Perform various checks on the dataset"""
    def __init__(self, df):
        self.df =df
        self.categorical_columns = [] 
        self.numerical_columns =[]
        self._identify_columns()

    def _identify_columns(self):
        """
        Identify numerical and categorical columns.
        """
        for col in self.df.columns:
            if self.df[col].dtype == object:
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
        
    def check_duplicates(self):
        """
        Displaying the duplicated rows for visual assesment
        """
        df_sorted = self.df.sort_values(by=self.df.columns.tolist())

        # Find duplicated rows
        duplicates = df_sorted[df_sorted.duplicated(keep=False)]

        print("***********************************************")
        print(" Total Number of Dulpicated Rows", len(duplicates))
        print("***********************************************")


        if not duplicates.empty:
            # Display the duplicated rows as HTML
            display(HTML(duplicates.to_html()))
        else:
            print("NO DUPLICATES FOUND")

    def drop_duplicated(self):
        """
        Upon confirmation of the duplicates. The duplicated data is dropped from the dataset
        """

        #Dropping the duplicates
        df = self.df.drop_duplicates(inplace= True)

        return df
       

    def check_missing(self):
        """
        Identify Null values in dataset as value count and percentage 
        """
        # Get features with null values
        null_features = self.df.columns[self.df.isnull().any()].tolist()
        
        if null_features:
            # Calculate the number of missing values for each feature
            null_counts = self.df[null_features].isnull().sum()
            
            # Calculate the percentage of missing data for each feature
            null_percentages = self.df[null_features].isnull().mean() * 100
            
            # Create a DataFrame to display the results
            null_info = pd.DataFrame({
                'Column Names': null_features,
                'Missing Values': null_counts,
                'Percentage Missing': null_percentages
            }).reset_index(drop=True)
            
            # Display the results
            display(null_info)
        else:
            print("NO NULL VALUES FOUND")       
    
    def check_outliers_and_plot(self):
        """
        Detect outliers in numerical columns using the IQR method and plot boxplots.
        """
        

        outlier_columns = []

        for column in self.numerical_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Find outliers
            outlier_indices = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)].index.tolist()

            if outlier_indices: 
                outlier_columns.append(column)

        print("***********************************************")
        print("Columns Containing Outliers Include:", outlier_columns)
        print("***********************************************")

        if outlier_columns:
            # Plot boxplots for columns with outliers
            num_rows = (len(outlier_columns) + 2) // 3
            num_cols = min(len(outlier_columns), 3)
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 8))

            # Ensure axes is always iterable
            if len(outlier_columns) == 1:
                axes = [axes]  # Make axes a list to allow consistent indexing
            else:
                axes = axes.flatten() if num_rows > 1 else axes

            for i, column in enumerate(outlier_columns):
                sns.boxplot(x=self.df[column], ax=axes[i])
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Values')
                axes[i].set_title(f'{column}')
                axes[i].tick_params(axis='x', rotation=45)

            # Remove any unused subplots
            if len(outlier_columns) < len(axes):
                for j in range(len(outlier_columns), len(axes)):
                    fig.delaxes(axes[j])

            # Adjust layout to prevent overlapping
            plt.tight_layout()
            plt.show()
        else:
            print("NO OUTLIERS FOUND")

    def convert_typing(self, convert_dict):
        # Iterate over the dictionary to convert columns
        for column_name, target_type in convert_dict.items():
            if target_type == 'datetime':  # Check for datetime conversion
                self.df[column_name] = pd.to_datetime(self.df[column_name])
                print(f"Column '{column_name}' has been successfully converted to datetime.")
            
            elif target_type == 'object':  # Check for object (string) conversion
                self.df[column_name] = self.df[column_name].astype("object")
                print(f"Column '{column_name}' has been successfully converted to object.")
            
            elif target_type == 'int':  # Check for integer conversion
                self.df[column_name] = self.df[column_name].astype("int64")
                print(f"Column '{column_name}' has been successfully converted to int64.")
            
            elif target_type == 'float':  # Check for float conversion
                self.df[column_name] = self.df[column_name].astype("float64")
                print(f"Column '{column_name}' has been successfully converted to float64.")
            
            else:
                print(f"Unsupported type '{target_type}' for column '{column_name}'.")

        # Return the updated DataFrame
        return self.df
    

class EDA:
    """ Classes to perform Univariate and Bivariate Analysis"""
    def __init__(self, df):
        self.df = df

    def plot_univariate_distribution(self, listed):
        # Create an empty list to hold statistics
        statistics_list = []
        
        plt.figure(figsize=(15, 10))
        
        # Flag to check if there are any numerical columns
        numerical_columns_found = False
        
        for i, column in enumerate(listed):
            plt.subplot(1, len(listed), i + 1)
            
            # Initialize a dictionary to store statistics for each column
            column_stats = {'Column': column}
            
            if self.df[column].dtype in ["int64", "float64"]:
                numerical_columns_found = True
                # For numerical columns, plot histogram and calculate statistics
                column_stats['Mean'] = self.df[column].mean()
                column_stats['Max'] = self.df[column].max()
                column_stats['Min'] = self.df[column].min()
                
                sns.histplot(self.df[column], kde=True, color='blue')
                plt.title(f"Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")

                # Append the statistics for this column to the list
                statistics_list.append(column_stats)
            
            elif self.df[column].dtype == "object":
                # For categorical columns, plot count plot with percentages
                value_counts = self.df[column].value_counts(normalize=True) * 100  # Normalize to get percentages
                sns.barplot(
                    x=value_counts.index, 
                    y=value_counts.values, 
                    hue=value_counts.index,
                    palette="viridis",
                )
                plt.title(f"Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel("Percentage")
                plt.xticks(rotation=45)
                
                # Annotate bars with percentages
                for index, value in enumerate(value_counts.values):
                    plt.text(index, value + 0.25, f"{value:.1f}%", ha="center", va="bottom")
        
        # Create the statistics DataFrame only if there are numerical columns
        if numerical_columns_found:
            statistics_df = pd.DataFrame(statistics_list).set_index("Column")
        else:
            # If no numerical columns, create an empty DataFrame
            statistics_df = pd.DataFrame(columns=["Column", "Mean", "Max", "Min"]).set_index("Column")
        
        # Display the DataFrame with statistics
        display(statistics_df)
        
        plt.tight_layout()
        plt.show()

    def plot_y_vs_numerical_columns(self,  x_list, y="Clicked_on_Ad",):
        for i, x in enumerate(x_list):
        # Create the boxplot
            fig = px.box(
                self.df, 
                y=y, 
                x=x,
                color="Clicked_on_Ad",  
                title=" Clicked on Ad vs " + x
            )
            
            fig.update_layout(legend=dict(title="Clicked on Ad", orientation="h", y=-0.2))  # Adjust legend position
            # fig.update_traces(quartilemethod="exclusive")
            fig.show()

    def plot_categorical_columns_vs_Clicked_Ads(self, x_list, hue='Clicked_on_Ad', palette="viridis"):
        """
        
        """
        plt.figure(figsize=(15, 8))  # Adjust the figure size to fit the plots

        for i, x in enumerate(x_list):
            # Check if the column exists in the dataframe
            if x in self.df.columns:
                plt.subplot(1, len(x_list), i + 1)  
                sns.countplot(data=self.df, x=x, hue=hue, palette=palette)
                plt.title(f'Distribution of {hue} vs {x}')
                plt.xticks(rotation=45) 
                
            else:
                print(f"Column '{x}' not found in the dataframe.")
        

        plt.tight_layout()  # To avoid overlap in subplot
        plt.show()

    def peak_period_for_ads_clicked(self, col, y='Clicked_on_Ad', ordered_hours=None, ordered_days=None):
        # Check if the column exists in the dataframe
        if col not in self.df.columns:
            print(f"Column '{col}' not found in the dataframe.")
            return
        
        # If ordered_hours or ordered_days is None, define them inside the method
        if ordered_hours is None:
            ordered_hours = [f"{i:02d}:00" for i in range(24)]
        if ordered_days is None:
            ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x=col, y=y, 
                    marker='X', color='purple', markersize=10, errorbar=None,
                    label='Click Through Rate')
        
        # Customizing the title and labels
        if col == 'Hour':
            plt.title("Click Through Rate by Hour of the Day")
            plt.xlabel('Hour of the Day')
            # Customize x-axis ticks for Hour (display all hours in order)
            plt.xticks(rotation=45)
            plt.xticks(ticks=range(len(ordered_hours)), labels=ordered_hours)
        elif col == 'Day':
            plt.title("Click Through Rate by Day of the Week")
            plt.xlabel('Day of the Week')
            # Customize x-axis ticks for Day (display days in order)
            plt.xticks(rotation=45)
            plt.xticks(ticks=range(len(ordered_days)), labels=ordered_days)

        # Limiting the y axis
        plt.ylim(0, 1)  

        # Calculating peak period (hour/day) and click-through rate
        peak_period = self.df.groupby(col, observed=False)[y].mean().idxmax()
        peak_value = self.df.groupby(col, observed=False)[y].mean().max()
        trough_period = self.df.groupby(col, observed=False)[y].mean().idxmin()
        trough_value = self.df.groupby(col, observed=False)[y].mean().min()
        
        print('*' * 80)
        print(f"PEAK {col.upper()} WAS AT {peak_period.upper()} WITH A CLICK THROUGH RATE OF {peak_value}")

        print(f"PEAK {col.upper()} WAS AT {trough_period.upper()} WITH A CLICK THROUGH RATE OF {trough_value}")

        print('*' * 80)
        print('\n')

        # Adding gridlines
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
        
        # Show the plot
        plt.tight_layout()
        plt.show()

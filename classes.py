# importing libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
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

class HypothesisTesting:
    def __init__(self, data):
        self.data = data
        self.results = pd.DataFrame(columns=['Hypothesis', 'Test', 'Statistic', 'P-Value', 'Result'])

    def add_result(self, hypothesis, test_name, statistic, p_value, result):
        "Add test result to the results DataFrame using pd.concat."
        result_df = pd.DataFrame({
            'Hypothesis': [hypothesis],
            'Test': [test_name],
            'Statistic': [statistic],
            'P-Value': [p_value],
            'Result': [result]
        })

        # Drop empty or NaN columns before concatenation
        result_df = result_df.loc[:, result_df.notna().any(axis=0)]
        
        # Concatenate the new result_df with the existing results DataFrame
        self.results = pd.concat([self.results, result_df], ignore_index=True)

    def test_age_vs_ad_clicks(self):
        """Test hypothesis for age vs. ad clicks."""
        corr, p_value = stats.pearsonr(self.data['Age'], self.data['Clicked_on_Ad'])

        result = "Reject H₀: Significant relationship between age and ad clicks." if p_value <0.005 else "Fail to reject H₀: No significant relationship between age and ad clicks."
        self.add_result('Age vs. Ad Clicks', 'Pearson Correlation', corr, p_value, result)
    


    def test_gender_vs_ad_clicks(self):
        """Test hypothesis for gender vs. ad clicks using Chi-Square Test."""
        # Create contingency table for 'Gender' and 'Clicked_on_Ad'
        contingency_table = pd.crosstab(self.data['Gender'], self.data['Clicked_on_Ad'])

        # Perform Chi-Square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Interpret results
        result = "Reject H₀: Significant relationship between gender and ad clicks." if p_value < 0.005 else "Fail to reject H₀: No significant relationship between gender and ad clicks."
        self.add_result('Gender vs. Ad Clicks', 'Chi-Square', chi2_stat, p_value, result)


    def test_area_income_vs_ad_clicks(self):
        """Test hypothesis for area income vs. ad clicks."""
        corr, p_value = stats.pearsonr(self.data['Area_Income'], self.data['Clicked_on_Ad'])

        result = "Reject H₀: Area income significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of area income on ad click likelihood."
        self.add_result('Area Income vs. Ad Clicks', 'Pearson Correlation', corr, p_value, result)

    def test_country_vs_ad_clicks(self):
        """Test hypothesis for country vs. ad clicks."""
        contingency = pd.crosstab(self.data['Country'], self.data['Clicked_on_Ad'])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

        result = "Reject H₀: Country significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of country on ad click likelihood."
        self.add_result('Country vs. Ad Clicks', 'Chi-Square', chi2_stat, p_value, result)

    def test_continent_vs_ad_clicks(self):
        """Test hypothesis for continent vs. ad clicks."""
        contingency = pd.crosstab(self.data['Continent'], self.data['Clicked_on_Ad'])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

        result = "Reject H₀: Continent significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of continent on ad click likelihood."
        self.add_result('Continent vs. Ad Clicks', 'Chi-Square', chi2_stat, p_value, result)

    def test_day_of_week_vs_ad_clicks(self):
        """Test hypothesis for day of the week vs. ad clicks."""
        anova_result = stats.f_oneway(
            self.data[self.data['Day'] == 'Monday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Tuesday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Wednesday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Thursday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Friday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Saturday']['Clicked_on_Ad'],
            self.data[self.data['Day'] == 'Sunday']['Clicked_on_Ad']
        )

        result = "Reject H₀: Ad click rate differs across days of the week." if anova_result.pvalue < 0.005 else "Fail to reject H₀: No significant difference in ad clicks across days of the week."
        self.add_result('Day of Week vs. Ad Clicks', 'ANOVA', anova_result.statistic, anova_result.pvalue, result)

    def test_time_of_day_vs_ad_clicks(self):
        """Test hypothesis for time of day (hour) vs. ad clicks."""
        anova_result = stats.f_oneway(
            self.data[self.data['Hour'] == '00:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '01:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '02:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '03:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '04:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '05:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '06:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '07:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '08:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '09:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '10:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '11:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '12:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '13:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '14:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '15:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '16:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '17:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '18:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '19:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '20:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '21:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '22:00']['Clicked_on_Ad'],
            self.data[self.data['Hour'] == '23:00']['Clicked_on_Ad']
        )

        result = "Reject H₀: The time of day significantly affects ad clicks." if anova_result.pvalue < 0.005 else "Fail to reject H₀: No significant effect of time of day on ad clicks."
        self.add_result('Time of Day vs. Ad Clicks', 'ANOVA', anova_result.statistic, anova_result.pvalue, result)

    def test_daily_internet_usage_vs_ad_clicks(self):
        """Test hypothesis for daily internet usage vs. ad clicks."""
        corr, p_value = stats.pearsonr(self.data['Daily_Internet_Usage'], self.data['Clicked_on_Ad'])

        result = "Reject H₀: Daily internet usage significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of daily internet usage on ad click likelihood."
        self.add_result('Daily Internet Usage vs. Ad Clicks', 'Pearson Correlation', corr, p_value, result)

    def test_daily_time_spent_vs_ad_clicks(self):
        """Test hypothesis for daily time spent on site vs. ad clicks."""
        corr, p_value = stats.pearsonr(self.data['Daily_Time_Spent_on_Site'], self.data['Clicked_on_Ad'])

        result = "Reject H₀: Daily time spent on site significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of daily time spent on site on ad click likelihood."
        self.add_result('Daily Time Spent vs. Ad Clicks', 'Pearson Correlation', corr, p_value, result)

    def test_ad_topic_line_vs_ad_clicks(self):
        """Test hypothesis for ad topic line vs. ad clicks."""
        contingency = pd.crosstab(self.data['Ad_Topic_Line'], self.data['Clicked_on_Ad'])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

        result = "Reject H₀: Ad topic line significantly affects ad click likelihood." if p_value < 0.005 else "Fail to reject H₀: No significant effect of ad topic line on ad click likelihood."
        self.add_result('Ad Topic Line vs. Ad Clicks', 'Chi-Square', chi2_stat, p_value, result)

    def get_results(self):
        """Return the results DataFrame."""
        return self.results



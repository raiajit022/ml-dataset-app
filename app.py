import os
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np
# Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 

def main():
	""" Common ML Dataset Explorer """
	st.title("Common ML Dataset Explorer")
	st.subheader("Datasets For ML Explorer with Streamlit")

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Streamlit is Awesome</p></div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	def file_selector(folder_path='./datasets'):
		"""
    Display a dropdown list of filenames in the specified folder and return the selected file.
    
    Parameters:
        folder_path (str): Path to the folder containing the files. Default is './datasets'.
        
    Returns:
        str: The selected file's full path.
    """
		filenames = os.listdir(folder_path) # Get a list of filenames in the specified folder
		selected_filename = st.selectbox("Select A file",filenames) # Display a dropdown list of filenames and return the selected file
		return os.path.join(folder_path,selected_filename) # Return the full path of the selected file

	filename = file_selector() # Select a file
	st.info("You Selected {}".format(filename)) # Display a message with the selected file

	# Read Data
	df = pd.read_csv(filename) # Read the CSV file into a Pandas DataFrame

	# Show Dataset

	if st.checkbox("Show Dataset"): # If the "Show Dataset" checkbox is checked
		st.dataframe(df.head(int(st.number_input("Number of Rows to View",min_value=1,step=1)))) # Display the data from the file as a Pandas DataFrame


	# Show Columns
	if st.button("Column Names"):  # If the "Column Names" button is clicked
		st.write(df.columns) # Display the column names of the DataFrame

	# Show Shape
	if st.checkbox("Shape of Dataset"):
		data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
		if data_dim == 'Rows':
			st.text("Number of Rows")
			st.write(df.shape[0])
		elif data_dim == 'Columns':
			st.text("Number of Columns")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Select Columns To Show"):
		all_columns = df.columns.tolist()
		selected_columns = st.multiselect("Select",all_columns)
		new_df = df[selected_columns]
		st.dataframe(new_df)
	
	# Show Values
	if st.button("Value Counts"):
		st.text("Value Counts By Target/Class")
		st.write(df.iloc[:,-1].value_counts())


	# Show Datatypes
	if st.button("Data Types"):
		st.write(df.dtypes)



	# Show Summary
	if st.checkbox("Summary"):
		st.write(df.describe().T)

	## Plot and Visualization

	st.subheader("Data Visualization")
	# Correlation
	# Seaborn Plot
	if st.checkbox("Correlation Plot[Seaborn]"):
		st.write(sns.heatmap(df.corr(),annot=True))
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()

	
	# Pie Chart
	if st.checkbox("Pie Plot"):
		all_columns_names = df.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.success("Generating A Pie Plot")
			st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

	# Count Plot
	if st.checkbox("Plot of Value Counts"):
		st.text("Value Counts By Target")
		all_columns_names = df.columns.tolist()
		primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
		selected_columns_names = st.multiselect("Select Columns",all_columns_names)
		if st.button("Plot"):
			st.text("Generate Plot")
			if selected_columns_names:
				vc_plot = df.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	# Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
	selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_plot == 'area':
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)

		elif type_of_plot == 'bar':
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)

		elif type_of_plot == 'line':
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)

		# Custom Plot 
		elif type_of_plot:
			cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

	if st.button("Thanks"):
		st.balloons()

	st.sidebar.header("About App")
	st.sidebar.info("A Simple EDA App for Exploring Common ML Dataset")

	st.sidebar.header("Get Datasets")
	st.sidebar.markdown("[Common ML Dataset Repo]("")")

	st.sidebar.header("About")
	st.sidebar.info("This app allows you to easily explore and visualize your data, helping you to gain insights and understand trends.")
	st.sidebar.text("Built with Streamlit")
	st.sidebar.text("Maintained by Ajit Rai(raiajit022@gmail.com)")


if __name__ == '__main__':
	main()
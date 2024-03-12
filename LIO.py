import os
# Get the absolute path of the directory containing the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the file path by joining the directory path and the filename
csv_file = os.path.join(script_directory, 'Table_summary.csv')

def app():
    import numpy as np
    import streamlit as st
    import pandas as pd

    st.title("Input - Output Model")
    st.markdown('This table has different sectors and their input as stated by [BEA Table 6.6B](https://apps.bea.gov/iTable/?reqid=150&step=3&isuri=1&table_list=6006&categories=io&_gl=1*1gckkur*_ga*NDUzMTk3OTQ4LjE3MDk0MTA2MTA.*_ga_J4698JNNFT*MTcwOTQxMDYwOS4xLjEuMTcwOTQxMTI2Ny42MC4wLjA.#eyJhcHBpZCI6MTUwLCJzdGVwcyI6WzEsMiwzXSwiZGF0YSI6W1sidGFibGVfbGlzdCIsIjYwMDkiXSxbImNhdGVnb3JpZXMiLCJHZHB4SW5kIl1dfQ==)')

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        st.write(df)
    except FileNotFoundError:
        st.error("File not found. Please ensure the file 'Table_summary.csv' is in the app folder.")

    # Define a function to calculate the Leontief Input-Output Model
    def calculate_leontief(coefficient, demand):
        coefficient_matrix_np = coefficient.to_numpy()
        final_demand_np = demand['Final demand'].to_numpy()

        total_output = np.linalg.inv(np.eye(coefficient.shape[0]) - coefficient).dot(demand)

        return total_output

    st.title("SELECTED SECTORS")
    with st.form(key='First Industry'):
        # Get the unique industry names from the DataFrame
        unique_names = df['Name'].unique()
        # Set default selections (replace 'Industry1' and 'Industry2' with your desired default values)
        default_selections = ['Farms','Forestry, fishing, and related activities']

        # Select the industries to work with
        rows = st.multiselect("Select the industries", unique_names, default=default_selections)

        # Add a button to trigger DataFrame creation
        col1, col2, col3 = st.columns(3)
        with col2:
            submit_button = st.form_submit_button(label='Create New DataFrame')

        # Define the new_df DataFrame outside of the button condition block
        new_df = pd.DataFrame()

        # Process data if button is clicked
        if submit_button:
            # Filter the DataFrame based on the selected rows
            filtered_df = df[df['Name'].isin(rows)]

            # Create a new DataFrame with the selected options as rows and columns
            new_df = pd.DataFrame(index=rows, columns=rows)

            # Iterate through the selected rows and columns, filling in the values in the new DataFrame
            for row in rows:
                for col in rows:
                    # Find the corresponding value in the filtered DataFrame
                    value_row = filtered_df[filtered_df['Name'] == row]
                    value_col = filtered_df[filtered_df['Name'] == col]
                    if not value_row.empty and not value_col.empty:
                        # Fill in the value in the new DataFrame
                        value = df.loc[value_row.index[0], col]  # Assuming the value is in the corresponding column
                        new_df.loc[row, col] = value

            # Calculate row totals and add a 'Total' column
            new_df['Total'] = new_df.sum(axis=1)

            # Calculate column totals and add a 'Total' row
            new_df.loc['Total'] = new_df.sum()

            # Display the new DataFrame
            st.write(new_df)

        # Define a function to process coefficient button click
        def process_coefficient_button(rows, df):
            filtered_df = df[df['Name'].isin(rows)]

            # Create a new DataFrame with the selected options as rows and columns
            new_df = pd.DataFrame(index=rows, columns=rows)

            # Iterate through the selected rows and columns, filling in the values in the new DataFrame
            for row in rows:
                for col in rows:
                    # Find the corresponding value in the filtered DataFrame
                    value_row = filtered_df[filtered_df['Name'] == row]
                    value_col = filtered_df[filtered_df['Name'] == col]
                    if not value_row.empty and not value_col.empty:
                        # Fill in the value in the new DataFrame
                        value = df.loc[value_row.index[0], col]  # Assuming the value is in the corresponding column
                        new_df.loc[row, col] = value

            # Calculate row totals and add a 'Total' column
            new_df['Total'] = new_df.sum(axis=1)

            # Calculate column totals and add a 'Total' row
            new_df.loc['Total'] = new_df.sum()

            # Extract the 'Total' column
            totals = new_df['Total']

            coefficient_matrix = new_df.copy()

            # Iterate through each column (excluding the 'Total' column)
            for idx, column in enumerate(new_df.columns[:-1]):
                # Divide each cell of the column by the corresponding cell of the Total column
                coefficient_matrix[column] = new_df[column] / totals.iloc[idx]

            # Remove the 'Total' column and row
            coefficient_matrix = coefficient_matrix.drop('Total', axis=1)
            coefficient_matrix = coefficient_matrix.drop('Total', axis=0)

            # Set both index name and column name to None to remove captions
            coefficient_matrix.columns.name = None
            coefficient_matrix.index.name = None

            return coefficient_matrix

        # Add a button to trigger DataFrame creation
        col1, col2, col3 = st.columns(3)
        with col2:
            coeff_button = st.form_submit_button(label='Coefficient')

        if coeff_button:
            # Call the function to process coefficient button click
            coefficient_matrix = process_coefficient_button(rows, df)
            st.write("Coefficient Matrix for Input-Output Model:")
            st.write(coefficient_matrix)
            st.write(coefficient_matrix.to_string(index=False, header=False))

    # Execute the coefficient code
    filtered_df = df[df['Name'].isin(rows)]
    new_df = pd.DataFrame(index=rows, columns=rows)

    for row in rows:
        for col in rows:
            value_row = filtered_df[filtered_df['Name'] == row]
            value_col = filtered_df[filtered_df['Name'] == col]
            if not value_row.empty and not value_col.empty:
                value = df.loc[value_row.index[0], col]
                new_df.loc[row, col] = value

    new_df['Total'] = new_df.sum(axis=1)
    totals = new_df['Total']
    coefficient_matrix = new_df.copy()

    for idx, column in enumerate(new_df.columns[:-1]):
        coefficient_matrix[column] = new_df[column] / totals.iloc[idx]

    coefficient_matrix = coefficient_matrix.drop('Total', axis=1)
    coefficient_matrix.columns.name = None
    coefficient_matrix.index.name = None

    # Get final demand input
    def get_final_demand(num_columns):
        final_demand = []
        for i in range(num_columns):
            demand = st.number_input(f"Enter final demand for column {i + 1}", value=0)
            final_demand.append(demand)
        return np.array(final_demand)

    num_columns = st.number_input("Enter the number of columns in the coefficient matrix", min_value=2, step=1)
    final_demand = get_final_demand(num_columns)
    # Ensure coefficient_matrix contains numeric values
    coefficient_matrix = coefficient_matrix.astype(float)

    # Convert coefficient_matrix to a NumPy array
    coefficient_matrix_np = coefficient_matrix.to_numpy()

    # Perform Leontief calculation
    c = np.eye(coefficient_matrix.shape[0])
    k = coefficient_matrix_np

    I = c
    # Changing our matrices to float
    F = k.astype(float)
    Q = I.astype(float)
    determinant = []
    for m in range(len(I)):
        row = []
        for n in range(len(I[0])):
            difference = round(Q[m][n] - F[m][n],4)
            row.append(difference)
        determinant.append(row)
    P = final_demand
    floated_P = P.astype(float)
    st.title("The TOTAL OUTPUT generated by each sector are:")
    t = np.linalg.inv(determinant)
    floated_t = t.astype(float)
    Result = np.round(floated_t.dot(floated_P),2)
    st.write(Result)
    # Write each sector's output level separately
    for i, output_level in enumerate(Result):
        st.write(f'The Output Level for {rows[i]} = {output_level}')
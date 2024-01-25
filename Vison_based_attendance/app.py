# Import necessary libraries
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Get the current timestamp and format it for date and timestamp
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Corrected file name concatenation
filename = f"Attendance_{date}.csv"  # Using the formatted date in the file name

# Assuming the file is generated with this name
df = pd.read_csv(filename)

# Use Streamlit to display the DataFrame in an interactive web app
st.dataframe(df.style.highlight_max(axis=0))


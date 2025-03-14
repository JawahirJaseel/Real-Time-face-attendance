import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Ensure streamlit_autorefresh is properly imported
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.error("The `streamlit_autorefresh` module is not installed. Install it using `pip install streamlit-autorefresh`.")
    st.stop()

# Get the current timestamp and formatted date/time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Auto-refresh setup
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# FizzBuzz logic
if count == 0:
    st.write("Count is Zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Load and display the attendance data
try:
    file_path = f"Attendance/Attendance_{date}.csv"
    df = pd.read_csv(file_path)
    st.dataframe(df.style.highlight_max(axis=0))
except FileNotFoundError:
    st.warning(f"Attendance file for today ({file_path}) not found. Please check the file path or create the file.")
except Exception as e:
    st.error(f"An error occurred while loading the attendance file: {e}")

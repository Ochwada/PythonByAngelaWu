import yfinance as yf
import pandas as pd
import datetime 
from datetime import date, timedelta

today = date.today()

date1 = today.strftime("%Y-%m-%d")
end_date = date1

date2 = date.today() -timedelta(days = 365)
date2 = date2.strftime("%Y-%m-%d")
start_date = date2 

data = yf.download ( 'AAPL', 
    start = start_date, 
    end = end_date,
    progress = False)

print(data[:10])


# Visualization 

import plotly.express as px 
fig = px.line(data, x = data.index, 
                y = "Close", 
                title = "Stock Price data")
fig.show()
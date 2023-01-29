#%%
import streamlit as st
import portfolyoulab as pl
import pandas as pd
from datetime import date, datetime
import numpy as np

last_year_end = date(date.today().year - 1, 12, 31)
n_rows = last_year_end.year - 1928 + 1 

db = pd.read_excel('histretSP.xlsx', sheet_name='Returns by year', skiprows=17, engine='openpyxl')

datetimeindex = pd.date_range('1928-12-31', last_year_end, freq='Y')

db = db[:n_rows]

columns = ['S&P 500 (includes dividends)', '3-month T.Bill', 'US T. Bond',\
   ' Baa Corporate Bond', 'Real Estate', 'Gold*', 'Inflation Rate', \
    'S&P 500 (includes dividends)2', '3-month T. Bill (Real)', '!0-year T.Bonds', \
    'Baa Corp Bonds', 'Real Estate3', 'Gold']

columns_names = ['S&P 500 TR', '3-month T.Bill', '10-year T.Bond', \
  'Baa Corporate Bond', 'Real Estate', 'Gold', 'Inflation Rate',\
  'S&P 500 TR (Real)', '3-month T.Bill (Real)', '10-year T.Bonds (Real)', \
  'Baa Corp Bonds (Real)', 'Real Estate (Real)', 'Gold (Real)']

db = db[columns]

db.columns = columns_names

db.index = datetimeindex

db.loc[pd.to_datetime('1927-12-31')] = [0] * len(columns_names)

db.sort_index(inplace=True)

db = db.astype('float')

options = [''] + columns_names

# Side Bar
with st.sidebar:

    # Datas
    start_date =  datetime.strptime('1927-12-31', '%Y-%m-%d') # Para ser desde 1927
    end_date =  datetime.strptime('2022-12-31', '%Y-%m-%d') # Para ser desde 1927
    start_date = st.date_input('Start Date', value=start_date, min_value=start_date, max_value=end_date) # O input
    end_date = st.date_input('End Date', value=end_date, min_value=start_date, max_value=end_date) # Input end Date, por defeito até hoje

    # Títulos
    title_1 = st.text_input(label='Chart 1 title', value='Performance')
    title_2 = st.text_input(label='Chart 2 title', value="Drawdowns")

    # Log y
    log_y = st.checkbox(label='Log Y', value=False)

    # Dropdowns selecção colunas
    asset_1 = st.selectbox(
    'Asset 1',
    options
    )
    asset_1_weight = st.slider(label='Asset 1 Weight (%)', min_value=0, max_value=100)

    if asset_1!='':
      options = [x for x in options if x != asset_1]

    asset_2 = st.selectbox(
    'Asset 2',
    options
    )
    asset_2_weight = st.slider(label='Asset 2 Weight (%)', min_value=0, max_value=100)

    if asset_2!='':
      options = [x for x in options if x != asset_2]

    asset_3 = st.selectbox(
    'Asset 3',
    options
    )
    asset_3_weight = st.slider(label='Asset 3 Weight (%)', min_value=0, max_value=100)

    if asset_3!='':
      options = [x for x in options if x != asset_3]

    asset_4 = st.selectbox(
    'Asset 4',
    options
    )
    asset_4_weight = st.slider(label='Asset 4 Weight (%)', min_value=0, max_value=100)

    st.markdown("**Nota:** O ouro só após o fim do acordo de Bretton Woods em 1971 começou a valorizar livremente no mercado!")

db = (db[start_date:end_date])

df = pd.DataFrame()

assets = [asset_1, asset_2, asset_3, asset_4]
assets_weight = [asset_1_weight, asset_2_weight, asset_3_weight, asset_4_weight]

df['assets'] = assets
df['assets_weight'] = assets_weight

df = df.loc[(df['assets'] != '')] 
df = df.loc[(df['assets_weight'] != 0)] 

db = db[df['assets']]

db = pl.compute_time_series(db)

weight = sum(df['assets_weight'])

if weight==100:
    portfolio = pl.compute_portfolio(db, df['assets_weight'])

    plot_performance = pl.ichart(portfolio, log_y=log_y, yticksuffix='$', yTitle='Valorização de investimento de 100 dólares', title=title_1, style='area')
    st.plotly_chart(plot_performance, use_container_width=False)

    plot_drawdowns = pl.ichart(pl.compute_drawdowns(portfolio), colors=['darkorange'], yticksuffix='%', yTitle='Abaixo de máximos', title=title_2, style='area')
    st.plotly_chart(plot_drawdowns, use_container_width=False)

    performance_table = pl.compute_performance_table(portfolio, freq='years')
    performance_table

    yearly_returns = pl.compute_yearly_returns(portfolio)
    yearly_returns
    st.markdown("**Nota:** Para períodos longo pode arrastar a tabela para a direita para ver todos os anos")

else:
    st.write('Weight not 100 or no asset selected')

# %%

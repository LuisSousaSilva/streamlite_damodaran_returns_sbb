
import numpy as np
import pandas as pd
import plotly.graph_objs as go

colors_list=['royalblue', 'darkorange',
           'dimgrey', 'rgb(86, 53, 171)',  'rgb(44, 160, 44)',
           'rgb(214, 39, 40)', '#ffd166', '#62959c', '#b5179e',
           'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
           'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
           'rgb(188, 189, 34)', 'rgb(23, 190, 207)'] * 10

def normalize(df):
    df = df.dropna()
    return (df / df.iloc[0]) * 100

def compute_time_series(dataframe, start_value=100):

#    INPUT: Dataframe of returns
#    OUTPUT: Growth time series starting in 100

    return (np.exp(np.log1p(dataframe).cumsum())) * start_value

def compute_drawdowns(dataframe):
    '''
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    '''
    return (dataframe / dataframe.cummax() -1) * 100

def compute_return(dataframe, years=''):
    '''
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    '''
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return (dataframe.iloc[-1] / dataframe.iloc[0] -1) * 100

    else:
        return (dataframe.iloc[-1] / dataframe.iloc[0] -1) * 100
    
def compute_max_DD(dataframe):
    return compute_drawdowns(dataframe).min()

def compute_cagr(dataframe, years=''):
    '''
    Function to calculate CAGR given a dataframe of prices
    '''
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return(dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)
    
    else:
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365
        
    return(dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)

def compute_mar(dataframe):
    '''
    Function to calculate mar: Return Over Maximum Drawdown
    given a dataframe of prices
    '''
    return compute_cagr(dataframe).div(compute_drawdowns(dataframe).min().abs())

def compute_StdDev(dataframe, freq='days'):    
    '''
    Function to calculate annualized standart deviation
    given a dataframe of prices. It takes into account the
    frequency of the data.
    '''    
    if freq == 'days':
        return dataframe.pct_change().std().mul((np.sqrt(252))).mul(100)
    if freq == 'months':
        return dataframe.pct_change().std().mul((np.sqrt(12))).mul(100)
    if freq == 'quarters':
        return dataframe.pct_change().std().mul((np.sqrt(4))).mul(100)
    if freq == 'years':
        return dataframe.pct_change().std().mul((np.sqrt(1))).mul(100)

def compute_yearly_returns(dataframe, start='1900', end='2100', style='table',
                        title='Yearly Returns', color=False, warning=True): 
    '''
    Style: table // string // chart
    '''
    # Getting start date
    start = str(dataframe.index[0])[0:10]

    # Resampling to yearly (business year)
    yearly_quotes = dataframe.resample('A').last()

    # Adding first quote (only if start is in the middle of the year)
    yearly_quotes = pd.concat([dataframe.iloc[:1], yearly_quotes])
    first_year = dataframe.index[0].year - 1
    last_year = dataframe.index[-1].year + 1

    # Returns
    yearly_returns = ((yearly_quotes / yearly_quotes.shift(1)) - 1) * 100
    yearly_returns = yearly_returns.set_index([list(range(first_year, last_year))])

    #### Inverter o sentido das rows no dataframe ####
    yearly_returns = yearly_returns.loc[first_year + 1:last_year].transpose()
    yearly_returns = round(yearly_returns, 2)

    # As strings and percentages
    yearly_returns.columns = yearly_returns.columns.map(str)    
    yearly_returns_numeric = yearly_returns.copy()

    if style=='table'and color==False:
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.format("{:.2%}")

    
    elif style=='table':
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.applymap(color_negative_red).format("{:.2%}")

    elif style=='numeric':
        yearly_returns = yearly_returns_numeric.copy()


    elif style=='string':
        for column in yearly_returns:
            yearly_returns[column] = yearly_returns[column].apply( lambda x : str(x) + '%')

        
    elif style=='chart':
        fig, ax = plt.subplots()
        fig.set_size_inches(yearly_returns_numeric.shape[1] * 1.25, yearly_returns_numeric.shape[0] + 0.5)
        yearly_returns = sns.heatmap(yearly_returns_numeric, annot=True, cmap="RdYlGn", linewidths=.2, fmt=".2f", cbar=False, center=0)
        for t in yearly_returns.texts: t.set_text(t.get_text() + "%")
        plt.title(title)
    
    else:
        print('At least one parameter has a wrong input')

    return yearly_returns

def compute_sharpe(dataframe, years='', freq='days'):   
    '''
    Function to calculate the sharpe ratio given a dataframe of prices.
    '''    
    return compute_cagr(dataframe, years).div(compute_StdDev(dataframe, freq))

def compute_performance_table(dataframe, years='si', freq='days'):
    '''
    Function to calculate a performance table given a dataframe of prices.
    Takes into account the frequency of the data.
    ''' 
    
    if years == 'si':
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365.25
        
        df = pd.DataFrame([compute_cagr(dataframe, years),
                           compute_StdDev(dataframe, freq),
                           compute_sharpe(dataframe, years, freq), compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']
        
        df = round(df.transpose(), 2)
        
        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
        # print_title('Performance from ' + start + ' to ' + end + ' (≈ ' + str(round(years, 1)) + ' years)')
        
        # Return object
        return df

    if years == 'ytd':
        last_year_end = dataframe.loc[str(last_year)].iloc[-1].name
        dataframe = dataframe[last_year_end:]

        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                    compute_StdDev(dataframe), compute_sharpe(dataframe),
                    compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)

        return df

    else:
        dataframe = filter_by_date(dataframe, years)
        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                           compute_StdDev(dataframe), compute_sharpe(dataframe),
                           compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
        
        if years == 1:
            print_title('Performance from ' + start + ' to ' + end + ' (' + str(years) + ' year)')
        else:
            print_title('Performance from ' + start + ' to ' + end + ' (' + str(years) + ' years)')
            
        return df

def ichart(data, title='', colors=colors_list, yTitle='', xTitle='', style='normal',
        width=990, height=500, hovermode='x', yticksuffix='', ytickprefix='',
        ytickformat="", source_text='', y_position_source='-0.125', xticksuffix='',
        xtickprefix='', xtickformat="", dd_range=[-50, 0], y_axis_range_range=None,
        log_y=False, image=''):

    '''
    style = normal, area, drawdowns_histogram
    colors = color_list or 
    hovermode = 'x', 'x unified', 'closest'
    y_position_source = -0.125 or bellow
    dd_range = [-50, 0]
    ytickformat =  ".1%"
    image: 'forum' ou 'fp
    
    '''
    
    if image=='fp':
        image='https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/FP-cor-positivo.png'
    elif image=='forum':
        image='https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/logo_forum.png'
        
    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor='#F5F6F9',
        plot_bgcolor='#F5F6F9',
        width=width,
        height=height,
        hovermode=hovermode,
        title=title,
        title_x=0.5,
        yaxis = dict(
            ticksuffix=yticksuffix,
            tickprefix=ytickprefix,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            range=y_axis_range_range,
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            title=yTitle,
            showgrid=True,
            tickformat=ytickformat,
                    ),
        xaxis = dict(
            title=xTitle,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            showgrid=True,
            tickformat=xtickformat,
            ticksuffix=xticksuffix,
            tickprefix=xtickprefix,
                    ),
        images= [dict(
            name= "watermark_1",
            source= image,
            xref= "paper",
            yref= "paper",
            x= -0.05500,
            y= 1.250,
            sizey= 0.20,
            sizex= 0.20,
            opacity= 1,
            layer= "below"
        )],

        margin=go.layout.Margin(
        l=50, #left margin
        r=75, #right margin
        b=50, #bottom margin
        t=100  #top margin
        ),

        annotations=[dict(
            xref="paper",
            yref="paper",
            x= 0.5,
            y= y_position_source,
            xanchor="center",
            yanchor="top",
            text=source_text,
            showarrow= False,
            font= dict(
                family="Arial",
                size=12,
                color="rgb(150,150,150)"
                )
        )
    ]

    ), # end

    if log_y:

        fig.update_yaxes(type="log")

    if style=='normal':
        z = -1
        
        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                mode='lines',
                name=i,
                line=dict(width=1.3,
                        color=colors[z]),
            ))

    if style=='area':
        z = -1
        
        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                hoverinfo='x+y',
                mode='lines',
                name=i,
                line=dict(width=0.7,
                        color=colors[z]),
                stackgroup='one' # define stack group
            ))

    if style=='drawdowns_histogram':
        fig.add_trace(go.Histogram(x=data.iloc[:, 0],
                     histnorm='probability',
                     marker=dict(colorscale='RdBu',
                                 reversescale=False,
                                 cmin=-24,
                                 cmax=0,
                                 color=np.arange(start=dd_range[0], stop=dd_range[1]),
                                 line=dict(color='white', width=0.2)),
                     opacity=0.75,
                     cumulative=dict(enabled=True)))

    return fig

def compute_portfolio(quotes, weights):
    
    Nomes=quotes.columns
    
    # Anos do Portfolio
    Years = quotes.index.year.unique()

    # Dicionário com Dataframes anuais das cotações dos quotes
    Years_dict = {}
    k = 0

    for Year in Years:
        # Dynamically create key
        key = Year
        # Calculate value
        value = quotes.loc[str(Year)]
        # Insert in dictionary
        Years_dict[key] = value
        # Counter
        k += 1

    # Dicionário com Dataframes anuais das cotações dos quotes
    Quotes_dict = {}
    Portfolio_dict = {}

    k = 0    
    
    for Year in Years:
        
        n = 0
        
        #Setting Portfolio to be a Global Variable
        global Portfolio
        
        # Dynamically create key
        key = Year

        # Calculate value
        if (Year-1) in Years:
            value = Years_dict[Year].append(Years_dict[Year-1].iloc[[-1]]).sort_index()
        else:
            value = Years_dict[Year].append(Years_dict[Year].iloc[[-1]]).sort_index()

        # Set beginning value to 100
        value = (value / value.iloc[0]) * 100
        # 
        for column in value.columns:
            value[column] = value[column] * weights[n]
            n +=1
        
        # Get Returns
        Returns = value.pct_change()
        # Calculating Portfolio Value
        value['Portfolio'] = value.sum(axis=1)

        # Creating Weights_EOP empty DataFrame
        Weights_EOP = pd.DataFrame()
        # Calculating End Of Period weights
        for Name in Nomes:
            Weights_EOP[Name] = value[Name] / value['Portfolio']
        # Calculating Beginning Of Period weights
        Weights_BOP = Weights_EOP.shift(periods=1)

        # Calculatins Portfolio Value
        Portfolio = pd.DataFrame(Weights_BOP.multiply(Returns).sum(axis=1))
        Portfolio.columns=['Simple']
        # Transformar os simple returns em log returns 
        Portfolio['Log'] = np.log(Portfolio['Simple'] + 1)
        # Cumsum() dos log returns para obter o preço do Portfolio 
        Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        Portfolio['Price'] = Portfolio['Price']   

        # Insert in dictionaries
        Quotes_dict[key] = value
        Portfolio_dict[key] = Portfolio
        # Counter
        k += 1

    # Making an empty Dataframe for Portfolio data
    Portfolio = pd.DataFrame()

    for Year in Years:
        Portfolio = pd.concat([Portfolio, Portfolio_dict[Year]['Log']])

    # Delete repeated index values in Portfolio    
    Portfolio.drop_duplicates(keep='last')

    # Naming the column of log returns 'Log'
    Portfolio.columns= ['Log']

    # Cumsum() dos log returns para obter o preço do Portfolio 
    Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        
    # Round Portfolio to 2 decimals and eliminate returns
    Portfolio = pd.DataFrame(round(Portfolio['Price'], 2))

    # Naming the column of Portfolio as 'Portfolio'
    Portfolio.columns= ['Portfolio']

    # Delete repeated days
    Portfolio = Portfolio.loc[~Portfolio.index.duplicated(keep='first')]

    return Portfolio
# %%

# Loading our Libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
#  Loading and Selecting Data
df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv', parse_dates=['Date'])
if df.empty:df.interpolate(inplace=True)
#print(df)
countries = ['Canada', 'Germany', 'United Kingdom', 'US', 'France', 'China']
df = df[df['Country'].isin(countries)]
#print(df)
# - Creating a Summary Column
#Selecting the data makes the resulting visualization a little more readable
df['Cases'] = df[['Confirmed', 'Recovered', 'Deaths']].sum(axis=1)
#print(df)
# Restructuring our Data by cases
df = df.pivot(index='Date', columns='Country', values='Cases')
#print(df)
countries = list(df.columns)
#print(countries)
covid = df.reset_index('Date')
#print(covid)
covid.set_index(['Date'], inplace=True)
# case by day
#print(covid)
covid.columns = countries; print(covid)

 #Calculating Rates per 100,000
populations = {'Canada':37664517, 'Germany': 83721496 , 'United Kingdom': 67802690 , 'US': 330548815, 'France': 65239883, 'China':1438027228}
percapita = covid.copy()
for country in list(percapita.columns):
    percapita[country] = percapita[country]/populations[country]*100000
print(percapita)

#  Generating Colours and Style
colors = {'Canada':'#045275', 'China':'#089099', 'France':'#7CCBA2', 'Germany':'#FCDE9C', 'US':'#DC3977', 'United Kingdom':'#7C1D6F'}
#the FiveThirtyEight style to add some general formatting
plt.style.use('fivethirtyeight')

#  Creating the Visualization
# plot = covid.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False)
# # the set_major_formatter method to format values with separators for thousands
# plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
# plot.grid(color='#d4d4d4')
# plot.set_xlabel('Date')
# plot.set_ylabel('# of Cases')

# #  Assigning Colour
# for country in list(colors.keys()):
#     plot.text(x = covid.index[-1], y = covid[country].max(), color = colors[country], s = country, weight = 'bold')
#     #print(covid.index[-1] , "s ", covid[country].max(), "s ", colors[country] )
# #  9 - Adding Labels
# plot.text(x = covid.index[1], y = int(covid.max().max()), s = "\nCOVID-19 Cases by Country\n\n", fontsize = 23, weight = 'bold', alpha = .75)
# plot.text(x = covid.index[1], y = int(covid.max().max()), s = "For the USA, China, Germany, France, United Kingdom, and Canada\nIncludes Current Cases, Recoveries, and Deaths", fontsize = 16, alpha = .75)
# print(covid.max())
percapitaplot = percapita.plot(figsize=(8,8), color=list(colors.values()), linewidth=5, legend=False)
percapitaplot.grid(color='#d4d4d4')
percapitaplot.set_xlabel('Date')
percapitaplot.set_ylabel('# of Cases per 100,000 People')
for country in list(colors.keys()):
    percapitaplot.text(x = percapita.index[-1], y = percapita[country].max(), color = colors[country], s = country, weight = 'bold')
percapitaplot.text(x = percapita.index[1], y = percapita.max().max()+25, s = "Per Capita COVID-19 Cases by Country\n\n", fontsize = 15, weight = 'bold', alpha = .75)
percapitaplot.text(x = percapita.index[1], y = percapita.max().max()+10, s = "For the USA, China, Germany, France, United Kingdom, and Canada\nIncludes Current Cases, Recoveries, and Deaths", fontsize = 16, alpha = .75)

plt.show()
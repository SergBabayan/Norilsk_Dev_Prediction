import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2030],
    'GDP_Growth_Rate': [
        100.0, 120.0, 140.0, 160.0, 180.0, 196.0, 197.0, 210.0, 220.0, 230.0, 239.1, 280.1
    ],  
    'Shipped_Goods_Volume': [
        334.3, 400.0, 500.0, 600.0, 700.0, 866.6, 945.2, 1000.0, 1100.0, 1200.0, 1245.3, 1503.2
    ], 
    'Employment_Count': [
        18044, 17800, 17600, 17400, 17200, 16046, 16190, 16300, 16500, 16700, 16859, 17869
    ], 
    'Investments': [
        500.0, 550.0, 600.0, 650.0, 700.0, 800.0, 850.0, 900.0, 1000.0, 1100.0, 1200.0, 1500.0
    ], 
    'Average_Salary': [
        50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000, 80000
    ], 
    'Unemployment_Rate': [
        6.5, 6.4, 6.3, 6.2, 6.1, 5.8, 5.7, 5.5, 5.4, 5.3, 5.2, 4.8
    ], 
    'Population': [
        175000, 176000, 177000, 178000, 179000, 180000, 181000, 182000, 183000, 184000, 185000, 190000
    ]  
}

df = pd.DataFrame(data)

X = df[['Shipped_Goods_Volume', 'Employment_Count', 'Investments', 'Average_Salary', 'Unemployment_Rate', 'Population']]  # Независимые переменные
y = df['GDP_Growth_Rate']  

X = sm.add_constant(X)

train_X = X[df['Year'] <= 2025]
train_y = y[df['Year'] <= 2025]

test_X = X[df['Year'] > 2025]
test_y = y[df['Year'] > 2025]

model = sm.OLS(train_y, train_X).fit()

print(model.summary())

predicted_y = model.predict(test_X)

mse = ((predicted_y - test_y) ** 2).mean()
print("\nMean Squared Error (MSE):", mse)


forecast_years = list(range(2026, 2036))
shipped_goods_growth_rate = (1503.2 - 1245.3) / (2030 - 2025)  
employment_growth_rate = (17869 - 16859) / (2030 - 2025)
investment_growth_rate = (1500.0 - 1200.0) / (2030 - 2025)
salary_growth_rate = (80000 - 70000) / (2030 - 2025)
unemployment_rate_change = (4.8 - 5.2) / (2030 - 2025)
population_growth_rate = (190000 - 185000) / (2030 - 2025)

forecast_data = {
    'Year': forecast_years,
    'Shipped_Goods_Volume': [1503.2 + shipped_goods_growth_rate * (year - 2030) for year in forecast_years],
    'Employment_Count': [17869 + employment_growth_rate * (year - 2030) for year in forecast_years],
    'Investments': [1500.0 + investment_growth_rate * (year - 2030) for year in forecast_years],
    'Average_Salary': [80000 + salary_growth_rate * (year - 2030) for year in forecast_years],
    'Unemployment_Rate': [4.8 + unemployment_rate_change * (year - 2030) for year in forecast_years],
    'Population': [190000 + population_growth_rate * (year - 2030) for year in forecast_years]
}

forecast_df = pd.DataFrame(forecast_data)

forecast_X = sm.add_constant(forecast_df[['Shipped_Goods_Volume', 'Employment_Count', 'Investments', 'Average_Salary', 'Unemployment_Rate', 'Population']])

forecast_y = model.predict(forecast_X)

forecast_df['Forecasted_GDP_Growth_Rate'] = forecast_y
print("\nПрогноз темпа роста ВМП до 2035 года:")
print(forecast_df)

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['GDP_Growth_Rate'], marker='o', linestyle='-', label='Исторические данные')
plt.plot(forecast_df['Year'], forecast_df['Forecasted_GDP_Growth_Rate'], marker='x', linestyle='--', label='Прогноз')
plt.title('Прогноз темпа роста ВМП до 2035 года (линейная регрессия)')
plt.xlabel('Год')
plt.ylabel('Темп роста (%)')
plt.legend()
plt.grid(True)

plt.savefig('gdp_growth_forecast.png', dpi=300, bbox_inches='tight')

plt.show()
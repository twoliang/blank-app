import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from matplotlib.dates import DateFormatter, DayLocator

# Notebooks content here

# Define functions or directly include the code from your notebooks
#读取文件
def read_and_process_data(file):
    df = pd.read_csv(file)
    
    # Ensure datetime column is correctly parsed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    return df

#输入时间范围
def select_one_month_data(df, start_date):
    # 确保 datetime 列是时间类型
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # 转换 start_date 为 pandas Timestamp 对象
    start_date = pd.Timestamp(start_date)
    
    # 计算 end_date，假设选择一个月的数据
    end_date = start_date + pd.DateOffset(months=1)
    
    # 使用 loc 方法选择时间范围
    selected_data = df.loc[start_date:end_date]
    
    return selected_data


def notebook1(df,start_date):

	# 提取目标变量和预测变量
# 选择一个月的数据
	one_month_data = select_one_month_data(df, start_date)

	X = one_month_data[['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN']]
	y_pm25 = one_month_data['PM2.5']
	y_pm10 = one_month_data['PM10']

	# 划分数据集
	X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
	_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)
 
 
	# 建立模型
	model_pm25 = LinearRegression()
	model_pm10 = LinearRegression()

	# 训练模型
	model_pm25.fit(X_train, y_pm25_train)
	model_pm10.fit(X_train, y_pm10_train)

	# 预测
	y_pm25_pred = model_pm25.predict(X_test)
	y_pm10_pred = model_pm10.predict(X_test)
 
 
 
	# 计算均方误差和决定系数
	mse_pm25 = mean_squared_error(y_pm25_test, y_pm25_pred)
	mse_pm10 = mean_squared_error(y_pm10_test, y_pm10_pred)
	r2_pm25 = r2_score(y_pm25_test, y_pm25_pred)
	r2_pm10 = r2_score(y_pm10_test, y_pm10_pred)

	# print(f"PM2.5 - Mean Squared Error (MSE): {mse_pm25}")
	# print(f"PM2.5 - R-squared (R²): {r2_pm25}")
	# print(f"PM10 - Mean Squared Error (MSE): {mse_pm10}")
	# print(f"PM10 - R-squared (R²): {r2_pm10}")
	st.write(f"PM2.5 - Mean Squared Error (MSE): {mse_pm25}")
	st.write(f"PM2.5 - R-squared (R²): {r2_pm25}")
	st.write(f"PM10 - Mean Squared Error (MSE): {mse_pm10}")
	st.write(f"PM10 - R-squared (R²): {r2_pm10}")
 
 
 
	# plt.figure(figsize=(10, 5))
	# plt.scatter(y_pm25_test, y_pm25_pred, color='blue', label='PM2.5 Predictions')
	# plt.scatter(y_pm10_test, y_pm10_pred, color='red', label='PM10 Predictions')
	# plt.xlabel('Actual Values')
	# plt.ylabel('Predicted Values')
	# plt.title('Actual vs Predicted Values')
	# plt.legend()
	# plt.grid(True)
	# plt.show()
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.scatter(y_pm25_test, y_pm25_pred, color='blue', label='PM2.5 Predictions')
	ax.scatter(y_pm10_test, y_pm10_pred, color='red', label='PM10 Predictions')
	ax.set_xlabel('Actual Values')
	ax.set_ylabel('Predicted Values')
	ax.set_title('Actual vs Predicted Values')
	ax.legend()
	ax.grid(True)
		
	st.pyplot(fig)
	
	# 添加截距项
	X = sm.add_constant(X)

	# 分割数据集为训练集和测试集
	X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
	_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)

	# 使用 statsmodels 进行回归分析
	model_pm25 = sm.OLS(y_pm25_train, X_train).fit()
	model_pm10 = sm.OLS(y_pm10_train, X_train).fit()

	# 输出回归结果
	# print(model_pm25.summary())
	# print(model_pm10.summary())
	st.write("各参数多元回归分析（PM2.5）结果:")
	st.write(model_pm25.summary())
	st.write("各参数多元回归分析（PM10）结果:")
	st.write(model_pm10.summary())



    
def notebook2(df, start_date):
    st.subheader('时间序列分析')
        # 选择一个月的数据
    df_month = select_one_month_data(df, start_date)
    
    # 确保日期时间列被正确解析和设置为索引
    if 'datetime' in df_month.columns:
        df_month['datetime'] = pd.to_datetime(df_month['datetime'])
        df_month.set_index('datetime', inplace=True)
    
    # 过滤掉包含非数值型数据的行（假设风向列名为'wd'）
    numerical_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df_month = df_month[numerical_columns].dropna()

    # 使用整个DataFrame作为训练集和测试集
    train = df_month.iloc[:-10]  # 前面的数据作为训练集
    test = df_month.iloc[-10:]   # 后面的10天数据作为测试集

    # 提取 PM2.5 列作为目标变量
    pm25_train = train['PM2.5']
    pm25_test = test['PM2.5']

    # 建立 ARIMA 模型
    final_model = ARIMA(pm25_train, order=(3, 0, 3))
    final_model_fit = final_model.fit()

    # 进行预测
    forecast = final_model_fit.forecast(steps=10)

    # 计算预测误差
    mse = mean_squared_error(pm25_test, forecast)
    rmse = np.sqrt(mse)
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # 绘制实际值和预测值的对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pm25_test.index, pm25_test, label='Actual PM2.5')
    ax.plot(pm25_test.index, forecast, label='Forecasted PM2.5', linestyle='--')
    ax.set_title('Comparison of Actual and Forecasted PM2.5')
    ax.set_xlabel('Date')
    ax.set_ylabel('PM2.5 Value')
    ax.legend()
    ax.grid(True)
        # 设置日期格式和间隔
    ax.xaxis.set_major_locator(DayLocator(interval=1))  # 每天一个间隔
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # 设置日期格式

    # 自动调整标签以避免重叠
    fig.autofmt_xdate()

    st.pyplot(fig)

    # 可选：计算并打印其他统计指标
    mae = mean_absolute_error(pm25_test, forecast)
    st.write(f"Mean Absolute Error (MAE): {mae}")




def notebook3(df, start_date):
	st.subheader('Anomaly Detection')
    # Content of third notebook
# 读取数据
	df_month = select_one_month_data(df, start_date)

	# 选择需要进行异常检测的字段
	columns_to_analyze = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN']

	# 定义函数来检测异常值
    # 定义函数来检测异常值
	def detect_outliers_zscore(data, threshold=3):
		outliers = pd.DataFrame()
		for col in data.columns:
			z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
			outliers[col] = z_scores > threshold
		return outliers

	# 检测异常值
	outliers = detect_outliers_zscore(df_month[columns_to_analyze])

	# 输出异常值的情况
	# print("异常值情况:")
	# print(outliers.sum())
	st.write("异常值情况:")
	st.write(outliers.sum())

	# 将异常值标记为True，非异常值标记为False
	df_month['is_outlier'] = outliers.any(axis=1)

	# 输出异常值数据
	# print("\n异常值数据示例:")
	# print(df[df['is_outlier']])
	st.write("\n异常值数据示例:")
	st.write(df_month[df_month['is_outlier']])
 
 
 
def notebook4(df, start_date):
	st.subheader('Environmental Data Analysis and Prediction')
    # Content of fourth notebook
	df_month = select_one_month_data(df, start_date)
	data = df_month
	# data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

	# 如果索引不是单调递增的，先对DataFrame进行排序
	if not data.index.is_monotonic_increasing:
		data = data.sort_index()



	# 确认选择的时间范围
	print("Min date in selected data:", data.index.min())
	print("Max date in selected data:", data.index.max())

	# 选择每天中午12点的数据
	data_daily_noon = data[data.index.hour == 12]
	# 选择每天中午12点的数据
	data_daily_noon = data[data.index.hour == 12]

	# 对TEMP进行ARIMA模型拟合示例
	model_temp = SARIMAX(data_daily_noon['TEMP'], order=(3,0,3)) 
	results_temp = model_temp.fit()

	# 对PRES进行ARIMA模型拟合示例
	model_pres = SARIMAX(data_daily_noon['PRES'], order=(3,0,3)) 
	results_pres = model_pres.fit()

	# 对DEWP进行ARIMA模型拟合示例
	model_dewp = SARIMAX(data_daily_noon['DEWP'], order=(3,0,3)) 
	results_dewp = model_dewp.fit()

	# 对WSPM进行ARIMA模型拟合示例
	model_wspm = SARIMAX(data_daily_noon['WSPM'], order=(3,0,3))  
	results_wspm = model_wspm.fit()

	# 预测未来10天的值，举例使用未来10个时间点（需根据你的时间频率进行调整）
	forecast_steps = 10
	forecast_temp = results_temp.forecast(steps=forecast_steps)
	forecast_pres = results_pres.forecast(steps=forecast_steps)
	forecast_dewp = results_dewp.forecast(steps=forecast_steps)
	forecast_wspm = results_wspm.forecast(steps=forecast_steps)

	# 输出预测结果示例
	print("Predicted TEMP:")
	print(forecast_temp)
	print("\nPredicted PRES:")
	print(forecast_pres)
	print("\nPredicted DEWP:")
	print(forecast_dewp)
	print("\nPredicted WSPM:")
	print(forecast_wspm)

	st.write("Predicted TEMP:")
	st.write(forecast_temp)
	st.write("\nPredicted PRES:")
	st.write(forecast_pres)
	st.write("\nPredicted DEWP:")
	st.write(forecast_dewp)
	st.write("\nPredicted WSPM:")
	st.write(forecast_wspm)

 
 
 
	# 从CSV文件加载数据，假设列名为PM2.5, PM10，时间列为datetime
	data = df_month
	# data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

	# 如果索引不是单调递增的，先对DataFrame进行排序
	if not data.index.is_monotonic_increasing:
		data = data.sort_index()



	# 选择每天中午12点的数据
	data_daily_noon = data[data.index.hour == 12]

	# 对PM2.5进行ARIMA模型拟合示例
	model_pm25_pre = SARIMAX(data_daily_noon['PM2.5'], order=(3, 0, 3))
	results_pm25_pre = model_pm25_pre.fit()

	# 对PM10进行ARIMA模型拟合示例
	model_pm10_pre = SARIMAX(data_daily_noon['PM10'], order=(3, 0, 3))
	results_pm10_pre = model_pm10_pre.fit()

	# 预测未来10天的值
	forecast_steps = 10
	forecast_pm25 = results_pm25_pre.forecast(steps=forecast_steps)
	forecast_pm10 = results_pm10_pre.forecast(steps=forecast_steps)

	# 输出预测结果示例
	print("Predicted PM2.5:")
	print(forecast_pm25)
	print("\nPredicted PM10:")
	print(forecast_pm10)

	st.write("Predicted PM2.5:")
	st.write(forecast_pm25)
	st.write("\nPredicted PM10:")
	st.write(forecast_pm10)

    
    
    
    
    
		# 示例数据集，假设有时间序列数据和北京市中心的经纬度
	data = pd.DataFrame({
		'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
		'latitude': [39.9042] * 10,  # 北京的纬度
		'longitude': [116.4074] * 10,  # 北京的经度
		'temp': forecast_temp.values,#传入预测的十天参数
		'pres': forecast_pres.values,
		'dewp': forecast_dewp.values,
		'wspm': forecast_wspm.values
	})

	# 计算污染物浓度（考虑温度、大气压力、露点和风速，并引入随机扰动）
	def gaussian_diffusion_with_noise(x, y, t, temp, pres, dewp, wspm):
		# 假设污染物浓度随时间指数衰减，空间高斯扩散，同时受温度、大气压力、露点和风速影响
		diffusion_coefficient = 1.0 + 0.1 * temp - 0.05 * pres + 0.05 * dewp + 0.2 * wspm  # 假设的扩散系数公式
		base_concentration = np.exp(-t) * np.exp(-diffusion_coefficient * ((x - 116.4074)**2 + (y - 39.9042)**2) / (2 * (1 + t)))
		noise = np.random.normal(loc=0, scale=5)  # 引入高斯白噪声，均值为0，标准差为5
		pollution_level = base_concentration + noise
		return np.abs(pollution_level)  # 取绝对值确保不会出现负数

	# 更新数据集，计算污染物浓度
	for i, row in data.iterrows():
		data.loc[i, 'pollution_level'] = gaussian_diffusion_with_noise(row['longitude'], row['latitude'], i,
																	row['temp'], row['pres'], row['dewp'], row['wspm'])

	# 创建动态地图
	fig = px.scatter_geo(data, 
						lat='latitude', 
						lon='longitude', 
						size='pollution_level', 
						color='pollution_level',
						animation_frame='timestamp',
						projection='natural earth',
						size_max=30,
						title='Pollution Diffusion Simulation with Noise in Beijing',
             			color_continuous_scale='Viridis')

	# 更新布局
	fig.update_geos(showcountries=True, countrycolor="DarkGray")
	fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})

	# 显示图表
	# fig.show()
	st.write("污染物扩散模拟图:")
	st.plotly_chart(fig)
 
 
 
 
	# 假设有预测出的PM2.5和PM10浓度数据，这里用示例数据代替
	forecast_steps = 10
	forecast_pm25 = np.random.uniform(low=5, high=50, size=forecast_steps)
	forecast_pm10 = np.random.uniform(low=10, high=160, size=forecast_steps)  # 假设最大值为160
	timestamps = pd.date_range(start='2024-07-11', periods=forecast_steps, freq='D')

	# 创建包含预测数据的DataFrame
	forecast_data = pd.DataFrame({
		'timestamp': timestamps,
		'PM2.5': forecast_pm25,
		'PM10': forecast_pm10
	})

	# 使用Plotly Express创建动态散点图
	fig = px.scatter(forecast_data, 
					x='timestamp', 
					y=['PM2.5', 'PM10'], 
					size_max=100,
					range_x=[timestamps.min(), timestamps.max()], 
					range_y=[0, forecast_pm10.max() + 10],  # 动态设置y轴范围，略高于最大预测值
					title='PM2.5 and PM10 Concentration Forecast',
					labels={'timestamp': 'Date', 'value': 'Concentration (µg/m³)'},
					template='plotly_dark',
					animation_frame='timestamp',
					color_discrete_sequence=['cyan', 'orange'])

	# 更新布局
	fig.update_layout(width=1000,
					height=800,
					xaxis_showgrid=False,
					yaxis_showgrid=False,
					paper_bgcolor='rgba(30, 30, 30, 1)',
					plot_bgcolor='rgba(30, 30, 30, 1)')

	# 显示图表
	# fig.show()
	st.write("污染物浓度预测图:")
	st.plotly_chart(fig)
		
    
    
    
    
    

def main():
    st.title('环境质量报告')
    st.markdown("<style>h1 {text-align: center;}</style>", unsafe_allow_html=True)

    # 用户输入数据文件路径
    data_file_path = st.text_input('请输入数据文件路径：')

    if data_file_path:
        # 读取并处理数据
        df = read_and_process_data(data_file_path)

        # 用户输入一个月的起始日期
        start_date = st.date_input('请选择一个月的起始日期：')

        if start_date:
            # 执行各个notebook部分
            st.header('多元回归线性分析')
            notebook1(df, start_date)

            st.header('时间序列分析')
            notebook2(df, start_date)

            st.header('环境质量异常监测')
            notebook3(df, start_date)

            st.header('污染物扩散模拟')
            notebook4(df, start_date)

if __name__ == "__main__":
    main()


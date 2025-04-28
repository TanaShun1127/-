import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
import numpy as np
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import re
import lightgbm as lgb
import altair as alt

#アプリタイトル
st.title("📈 株価予測アプリ")

st.sidebar.write("""詳細情報""")
# サイドバー：ユーザー入力
ticker = st.sidebar.text_input("ティッカーシンボルを入力してください", "AAPL")
start_date = st.sidebar.date_input("開始日", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("終了日", datetime.today())

def get_stock_yf(stock,start,end):
    df = yf.download(tickers=stock, start=start, end=end)
    return df

df = pd.DataFrame()

# 株価データの取得
data = get_stock_yf(ticker, start_date, end_date)
# カラム名を単一に変更する
data.columns = data.columns.map('_'.join)

# 特徴量とターゲットの作成
# ここでは終値を予測するために、終値を1日シフトさせたものをターゲットとする
data['Target'] = data[f'Close_{ticker}'].shift(-1)
features = [ f'Open_{ticker}',f'High_{ticker}',f'Low_{ticker}',f'Close_{ticker}',f'Volume_{ticker}'] # 'Open','High', 'Low','Close'

# データセットの分割
X = data[features].iloc[:-1]  # 最後の行はターゲットがNaNなので除外
y = data['Target'].iloc[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#XGBoostで学習するためのデータ形式に変換
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_test, y_test)

#モデルパラメータの設定
params = {'metric' : 'rmse'}
# トレーニングデータから特徴名を取得する
# # マルチインデックスを文字列の列名を持つ単一レベルインデックスに変換する
# X_train.columns = X_train.columns.map('_'.join)
feature_names = X_train.columns.tolist()

# Pass feature names to the model
model = lgb.train(params, dtrain, feature_name=feature_names)

# 予測と評価
predictions = model.predict(X_test)
df['predictions'] = predictions
df['Date'] = y_test.index
df_ = df.sort_values('Date', ascending = True)

# 予測と評価
predictions = model.predict(X_test)
df['predictions'] = predictions
df['Date'] = y_test.index
df_ = df.sort_values('Date', ascending = True)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

y_test_ = y_test.sort_index(ascending=True)

# データフレームの準備
plot_df = pd.DataFrame({
    'Date': df_['Date'],
    'Actual': y_test_.values.flatten(),  # y_testがSeriesなら .values だけでOK
    'Predicted': df_['predictions'].values
})

# 長い（tidy）形式に変換
plot_df_long = plot_df.melt('Date', var_name='Type', value_name='Value')

# Altairプロット
chart = alt.Chart(plot_df_long).mark_line().encode(
    x='Date:T',
    y='Value:Q',
    color='Type:N',
    tooltip=['Date:T', 'Type:N', 'Value:Q']
).properties(
    width=700,
    height=400,
    title='Actual vs Predicted'
).interactive()

# Streamlitで表示
st.altair_chart(chart, use_container_width=True)
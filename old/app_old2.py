import yfinance as yf
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time  # ←ここを追加！

# アプリタイトル
st.title("📈 株価予測アプリ")

# サイドバー
st.sidebar.write("""## 詳細情報""")
ticker = st.sidebar.text_input("ティッカーシンボルを入力してください", "AAPL")
start_date = st.sidebar.date_input("開始日", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("終了日", datetime.today())

def get_stock_yf(stock, start, end):
    df = yf.download(tickers=stock, start=start, end=end)
    return df

if st.button('予測する'):

    # プログレスバーを作成
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = get_stock_yf(ticker, start_date, end_date)
    data.columns = data.columns.map('_'.join)

    data['Target'] = data[f'Close_{ticker}'].shift(-1)
    features = [f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Volume_{ticker}']

    X = data[features].iloc[:-1]
    y = data['Target'].iloc[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = lgb.Dataset(X_train, y_train)
    params = {'metric': 'rmse'}
    feature_names = X_train.columns.tolist()
    
    # 少し進める
    progress_bar.progress(20)
    status_text.text('学習中...')

    model = lgb.train(params, dtrain, feature_name=feature_names)

    progress_bar.progress(50)
    status_text.text('過去データの予測中...')

    predictions = model.predict(X_test)

    # 5営業日先の予測
    last_features = X.iloc[-1].values.reshape(1, -1)
    future_preds = []
    future_dates = []

    current_date = data.index[-1]

    for i in range(5):
        pred_price = model.predict(last_features)[0]
        future_preds.append(pred_price)
        current_date += pd.tseries.offsets.BDay(1)
        future_dates.append(current_date)

        # 次の予測のために更新
        last_features[0][features.index(f'Open_{ticker}')] = pred_price
        last_features[0][features.index(f'High_{ticker}')] = pred_price
        last_features[0][features.index(f'Low_{ticker}')] = pred_price
        last_features[0][features.index(f'Close_{ticker}')] = pred_price

        # プログレスバーを少しずつ進める
        progress_bar.progress(50 + (i+1)*10)
        time.sleep(0.3)  # 少しだけ待つと「進んでる感」演出できる

    # 可視化用データ準備
    history_actual = pd.DataFrame({
        'Date': X_test.index,
        'Type': 'Actual',
        'Value': y_test.values
    })

    history_pred = pd.DataFrame({
        'Date': X_test.index,
        'Type': 'Predicted',
        'Value': predictions
    })

    future_pred = pd.DataFrame({
        'Date': future_dates,
        'Type': 'Future_Predicted',
        'Value': future_preds
    })

    plot_df = pd.concat([history_actual, history_pred, future_pred])

    # グラフ描画
    base = alt.Chart(plot_df).encode(
        x='Date:T',
        y='Value:Q',
        color=alt.Color('Type:N', scale=alt.Scale(
            domain=['Actual', 'Predicted', 'Future_Predicted'],
            range=['blue', 'red', 'orange']
        )),
    )

    line_actual = base.transform_filter(
        alt.datum.Type == 'Actual'
    ).mark_line()

    line_pred = base.transform_filter(
        alt.datum.Type == 'Predicted'
    ).mark_line()

    line_future = base.transform_filter(
        alt.datum.Type == 'Future_Predicted'
    ).mark_line(strokeDash=[5,5])  # ←オレンジの点線に！

    chart = (line_actual + line_pred + line_future).properties(
        width=700,
        height=400,
        title='Actual vs Predicted (+5営業日 Future Prediction)'
    ).interactive()

    # プログレスバー完了
    progress_bar.progress(100)
    status_text.text('完了！')

    # グラフ表示
    st.altair_chart(chart, use_container_width=True)

    # 未来予測結果をテーブル表示
    st.subheader("📄 5営業日分の予測結果")
    future_display = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_preds
    })
    st.dataframe(future_display)

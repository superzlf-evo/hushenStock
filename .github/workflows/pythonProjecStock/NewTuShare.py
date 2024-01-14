import tushare as ts
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, request
from mpmath import mpf
import mplfinance as mpf

matplotlib.use("Agg")
# 正确显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种常见的中文黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# pro = ts.pro_api('9f908fee26955d11a2c2b335f1e9e75759021bbdbb9a4abf8d241e4b')
# pro = ts.pro_api('c51b1a1e12b413dfb84c1979442fce5044f1ced65859fd8f2bca9446')
pro = ts.pro_api("ea256de2b216f34ba775d052c44383d417646a86df6032a8bdb32896")


# 输入股票代码获取该股票今年所有股市信息，写入到Stock库中Stock表内
def Get_Stock(stock_id):
    # 拉取数据
    end_date = datetime.now()
    # start_date = end_date - timedelta(days=100)

    # 格式化日期
    end_date_str = end_date.strftime("%Y%m%d")
    # start_date_str = start_date.strftime("%Y%m%d")
    df = pro.daily(
        **{
            "ts_code": stock_id,
            "trade_date": "",
            "start_date": "20230101",
            "end_date": end_date_str,
            "offset": "",
            "limit": "",
        },
        fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ],
    )
    # print(df)
    # df.to_csv('2023Stock.csv', index=False)
    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)
    try:
        df.to_sql("Stock", conn, if_exists="replace", index=False)
    except Exception as e:
        print(f"Error writing to database: {e}")
    conn.close()
    return df


##########################################################


def Get_Stock_List(keyword):
    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)
    # 检查stock_basic.csv文件是否存在
    if not os.path.exists("stock_basic.csv"):
        df = pro.stock_basic(
            **{
                "ts_code": "",
                "name": "",
                "exchange": "",
                "market": "",
                "list_status": "",
                "is_hs": "",
                "limit": "",
                "offset": "",
            },
            fields=[
                "ts_code",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "list_date",
            ],
        )
        df.to_csv("stock_basic.csv", index=False)
        df.to_sql("Stock_List", conn, if_exists="replace", index=False)
    else:
        # 从数据库中读取数据
        df = pd.read_sql("SELECT * FROM Stock_List", conn)
    # 执行SQL查询
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Stock_List WHERE name LIKE '%{}%'".format(keyword))
    rows = cursor.fetchall()

    # 如果查询结果为空，则返回一个空的DataFrame
    if not rows:
        print("没有找到带有'{}'的相关信息".format(keyword))
        # return pd.DataFrame()
    else:
        # 将查询结果转换为DataFrame
        return pd.DataFrame(
            rows,
            columns=[
                "ts_code",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "list_date",
            ],
        )

    # 关闭游标和连接
    cursor.close()
    conn.close()


# 获取所有沪深股票公司的相关内部信息，若已经有表则不再抓取，所有数据写入到Stock库中Basic_infor表内
def Basic_information():
    if os.path.exists("Basic_infor.csv"):
        db_file = "Stock.db"
        conn = sqlite3.connect(db_file)
    else:
        df = pro.stock_company(
            **{"ts_code": "", "exchange": "", "status": "", "limit": "", "offset": ""},
            fields=[
                "ts_code",
                "exchange",
                "chairman",
                "manager",
                "secretary",
                "reg_capital",
                "setup_date",
                "province",
                "city",
                "website",
                "email",
                "employees",
                "main_business",
                "business_scope",
                "ann_date",
                "office",
                "introduction",
            ],
        )
        # print(df)
        df.to_csv("Basic_infor.csv", index=False)
        db_file = "Stock.db"
        conn = sqlite3.connect(db_file)
        df.to_sql("Basic_infor", conn, if_exists="replace", index=False)
    # cursor = conn.cursor()
    # cursor.execute("SELECT * FROM Basic_infor")
    # rows = cursor.fetchall()
    # for row in rows:
    #     print(row)
    # cursor.close()
    conn.close()


# 通过股票代码获取该股票的公司内部信息，用于前端展示
def get_company_info(stock_code):
    # 连接到数据库
    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # 执行查询语句
        cursor.execute("SELECT * FROM Basic_infor WHERE ts_code = ?", (stock_code,))
        # 获取查询结果
        row = cursor.fetchone()

        if row:
            # 将查询结果转换为字典格式（可选）
            info = {
                "ts_code": row[0],
                "exchange": row[1],
                "chairman": row[2],
                "manager": row[3],
                "secretary": row[4],
                "reg_capital": row[5],
                "setup_date": row[6],
                "province": row[7],
                "city": row[8],
                "introduction": row[9],
                "website": row[10],
                "email": row[11],
                "office": row[12],
                "ann_date": row[13],
                "business_scope": row[14],
                "employees": row[15],
                "main_business": row[16],
            }
            return info
        else:
            return None  # 或返回一个合适的消息，如 '股票代码未找到'

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()


# 计算XX天的均线
def Get_roll_dMean(num):
    conn = sqlite3.connect("Stock.db")
    # 创建一个游标对象
    cursor = conn.cursor()

    data = cursor.execute(
        "SELECT close, trade_date FROM Stock ORDER BY trade_date ASC"
    ).fetchall()

    # 将数据转换为pandas DataFrame
    df = pd.DataFrame(data, columns=["close", "date"])
    # 将日期列转换为 datetime 对象，以便进行日期操作
    df["date"] = pd.to_datetime(df["date"])
    # 创建5天移动平均线
    df["dMean"] = df["close"].rolling(num).mean()
    # 仅保留在最近100天内的数据，并按照日期排序
    df1 = df.iloc[-100:, :].sort_values(by="date", ascending=True)

    cursor.close()
    conn.close()
    return df1


# 获取金叉死叉，即投资抛出的时间节点，导出表Death_cross和Golden_cross
def GetGD_x():  # 金叉，死叉展示
    df1 = Get_roll_dMean(5)
    df2 = Get_roll_dMean(30)
    df1.set_index("date", inplace=True)
    df2.set_index("date", inplace=True)

    d5_Mean = df1["dMean"]
    d30_Mean = df2["dMean"]
    plt.figure(1)
    plt.plot(d5_Mean.iloc[-100:], label="5dMean")
    plt.plot(d30_Mean.iloc[-100:], label="30dMean")
    # plt.show()
    plt.legend()
    # plt.savefig("static/images/GD_cha.png", dpi=300)
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_path, "static", "images", "GD_cha.png")
    plt.savefig(file_path, dpi=300)

    df = df1
    df = df.rename(columns={"dMean": "5dMean"})
    df["30dMean"] = df2["dMean"]
    sr1 = df["5dMean"] < df["30dMean"]
    sr2 = df["5dMean"] >= df["30dMean"]

    death_cross = df[sr1 & sr2.shift(1)]
    death_cross=death_cross.iloc[-100:,:]
    golden_cross = df[~(sr1 | sr2.shift(1))]
    golden_cross = golden_cross.iloc[-100:, :]
    if len(death_cross) > 0:
        print("最近合适抛出股票的时间：")
        print(death_cross)
    else:
        print("最近合适抛出股票的时间！！！")
    if len(golden_cross) > 0:
        print("最近合适投入股票的时间：")
        print(golden_cross)
    else:
        print("最近没有合适投入股票的时间！！！")
    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)
    death_cross.to_sql("Death_cross", conn, if_exists="replace", index=True)
    golden_cross.to_sql("Golden_cross", conn, if_exists="replace", index=True)
    conn.close()


# 获取后30股票日的金叉死叉，即投资抛出的时间节点，使用ARIMA模型预测，取得预测值，生成表PR_Death_cross和PR_Golden_cross
def Get_Future():
    # 加载数据
    conn = sqlite3.connect("Stock.db")
    cursor = conn.cursor()
    data = cursor.execute(
        "SELECT  trade_date ,close  FROM Stock ORDER BY trade_date ASC"
    ).fetchall()
    data = pd.DataFrame(data, columns=["date", "close"])
    data = data.iloc[-99:, :]
    # 将日期列转换为 datetime 对象，以便进行日期操作
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)
    data = data["close"].values
    data = data.reshape(-1, 1)
    cursor.close()
    conn.close()

    # 定义ARIMA模型并训练模型
    model = ARIMA(data, order=(3, 2, 0))  # 参数配置，如p、d、q等。
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(data), end=len(data) + 30)  # 预测未来5天的收盘价。
    rmse = np.sqrt(
        mean_squared_error(predictions, data[len(data) - 31 :])
    )  # 计算预测的RMSE值。
    data = data.ravel()
    array_combined = np.concatenate((data, predictions))
    newdata = pd.DataFrame(array_combined, columns=["close"])
    # newdata=newdata.reset_index()
    # print(newdata)
    newdata["dMean_5"] = newdata["close"].rolling(5).mean()
    newdata["dMean_30"] = newdata["close"].rolling(30).mean()
    df1 = newdata
    # print(df1)
    x = np.arange(1, 31)
    plt.figure(2)
    plt.plot(x, df1.iloc[-30:, -2], label="5dMean")
    plt.plot(x, df1.iloc[-30:, -1], label="30dMean")
    # plt.show()
    plt.legend()
    # plt.savefig("static/images/Future_cha.png", dpi=300)
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_path, "static", "images", "Future_cha.png")

    plt.savefig(file_path, dpi=300)

    sr1 = df1["dMean_5"] < df1["dMean_30"]
    sr2 = df1["dMean_5"] > df1["dMean_30"]
    df1 = df1.reset_index()
    # print(df1)
    death_cross = df1[sr1 & sr2.shift(1)]
    death_cross = death_cross[death_cross["index"] > 99]
    golden_cross = df1[~(sr1 | sr2.shift(1))]
    golden_cross = golden_cross[golden_cross["index"] > 99]
    for i in range(len(death_cross)):
        death_cross.iloc[i, 0] = death_cross.iloc[i, 0] - 99
    for i in range(len(golden_cross)):
        golden_cross.iloc[i, 0] = golden_cross.iloc[i, 0] - 99
    death_cross = death_cross.rename(
        columns={
            "index": "future",
            "close": "close",
            "dMean_5": "dMean_5",
            "dMean_30": "dMean_30",
        }
    )
    golden_cross = golden_cross.rename(
        columns={
            "index": "future",
            "close": "close",
            "dMean_5": "dMean_5",
            "dMean_30": "dMean_30",
        }
    )

    if len(death_cross) > 0:
        print("预测30股票日内合适抛出股票的时间：")
        print(death_cross)
    else:
        print("预测30股票日内没有合适抛出股票的时间！！！")
    if len(golden_cross) > 0:
        print("预测30股票日内合适投入股票的时间：")
        print(golden_cross)
    else:
        print("预测30股票日内没有合适投入股票的时间！！！")

    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)
    death_cross.to_sql("PR_Death_cross", conn, if_exists="replace", index=False)
    golden_cross.to_sql("PR_Golden_cross", conn, if_exists="replace", index=False)
    conn.close()


def plot_volume_amount_and_pct_change():
    db_file = "Stock.db"
    conn = sqlite3.connect(db_file)

    # 从数据库中读取成交量、成交额和涨跌幅度
    query = "SELECT trade_date, vol, amount, pct_chg FROM Stock WHERE trade_date >= '20230901'"
    df = pd.read_sql(query, conn)
    conn.close()

    # 确保日期格式正确，并设置为索引
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)

    # 创建成交量的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(df["vol"], label="成交量", color="blue")
    plt.title("成交量")
    plt.xlabel("日期")
    plt.ylabel("成交量")
    plt.legend()

    # 确保 static/images 目录存在
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    # 保存成交量的折线图
    plt.savefig("static/images/volume.png")
    plt.close()

    # 创建成交额的柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df["amount"], label="成交额", color="green")
    plt.title("成交额")
    plt.xlabel("日期")
    plt.ylabel("成交额")
    plt.legend()

    # 保存成交额的柱状图
    plt.savefig("static/images/amount.png")
    plt.close()

    # 确保 static/images 目录存在
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    # 创建涨跌幅度的图表
    plt.figure(figsize=(10, 6))
    plt.plot(df["pct_chg"], label="涨跌幅度", color="red")
    plt.title("涨跌幅度")
    plt.xlabel("日期")
    plt.ylabel("百分比 (%)")
    plt.legend()

    # 保存涨跌幅度图表
    plt.savefig("static/images/pct_change.png")
    plt.close()


def GetDailyCandlestickChart():  # 绘制日线k图，并返回买入卖出建议
    # 连接数据库
    conn = sqlite3.connect("Stock.db")
    cursor = conn.cursor()

    # 获取数据
    data = cursor.execute(
        "SELECT open, close, low, high, trade_date FROM Stock ORDER BY trade_date ASC"
    ).fetchall()
    df = pd.DataFrame(data, columns=["open", "close", "low", "high", "date"])

    # 转换日期并设置索引
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 使用日线数据
    daily_df = df

    # 计算SMA和RSI
    daily_df["SMA"] = calculate_sma(daily_df["close"])
    daily_df["RSI"] = calculate_rsi(daily_df["close"])

    # 趋势分析
    trend_description = ""
    if (
        daily_df["close"].iloc[-1] > daily_df["SMA"].iloc[-1]
        and daily_df["RSI"].iloc[-1] < 70
    ):
        trend_description = "上升趋势 - 考虑买入"
    elif (
        daily_df["close"].iloc[-1] < daily_df["SMA"].iloc[-1]
        and daily_df["RSI"].iloc[-1] > 30
    ):
        trend_description = "下降趋势 - 考虑卖出"
    else:
        trend_description = "横盘市场 - 没有明确信号"

    # 绘制日线K线图
    mpf.plot(
        daily_df,
        type="candle",
        style="yahoo",
        title="Daily Candlestick Chart",
        savefig="static/images/daily_candlestick_chart.png",
    )
    # 打印趋势描述
    return trend_description


def calculate_sma(data, window=10):  ###sma指数
    return data.rolling(window).mean()


# 计算相对强弱指数 (RSI)
def calculate_rsi(data, window=14):  ###rsi 指数
    delta = data.diff()
    up_changes = delta.clip(lower=0)
    down_changes = -1 * delta.clip(upper=0)

    up_changes_avg = up_changes.rolling(window).mean()
    down_changes_avg = down_changes.rolling(window).mean()

    rs = up_changes_avg / down_changes_avg
    rsi = 100 - (100 / (1 + rs))
    return rsi


def GetMonthlyCandlestickChart():  # 月k线图
    # 连接数据库
    conn = sqlite3.connect("Stock.db")
    cursor = conn.cursor()

    # 获取数据
    data = cursor.execute(
        "SELECT open, close, low, high, trade_date FROM Stock ORDER BY trade_date ASC"
    ).fetchall()
    df = pd.DataFrame(data, columns=["open", "close", "low", "high", "date"])

    # 转换日期并设置索引
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 重采样数据到月度
    monthly_df = df.resample("M").agg(
        {"open": "first", "close": "last", "low": "min", "high": "max"}
    )

    # 计算SMA和RSI
    monthly_df["SMA"] = calculate_sma(monthly_df["close"])
    monthly_df["RSI"] = calculate_rsi(monthly_df["close"])
    # 绘制月线K线图
    mpf.plot(
        monthly_df,
        type="candle",
        style="yahoo",
        title="Monthly Candlestick Chart",
        savefig="static/images/monthly_candlestick_chart.png",
    )


# 绘制ATR图
def calculate_monthly_atr_sum(df, window=14):  # 计算ATR的数值，供Get_ATR（）调用
    """Calculate Monthly Sum of Average True Range (ATR)"""
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=window).mean()
    monthly_atr_sum = df["atr"].resample("MS").sum()
    return monthly_atr_sum


def plot_monthly_atr(monthly_atr_sum):  # 绘制ATR柱状图供Get_ATR（）调用
    """Plot Monthly ATR Sum"""
    plt.figure(figsize=(15, 10))
    plt.bar(monthly_atr_sum.index, monthly_atr_sum, width=20, label="Monthly ATR Sum")
    plt.title(" Monthly Sum of Average True Range (ATR)")
    plt.xlabel("Month")
    plt.ylabel("ATR Sum")
    plt.legend()
    # plt.show()
    save_path = "static/images/Month_ATR.png"
    plt.savefig(save_path, dpi=600)
    plt.close()


def Get_ATR():  # 这是ATR的主方法体，用这个
    # 连接数据库并检索数据
    conn = sqlite3.connect("Stock.db")
    cursor = conn.cursor()
    query = "SELECT trade_date, high, low, close FROM Stock where trade_date<20231201 ORDER BY trade_date ASC"
    data = cursor.execute(query).fetchall()
    df = pd.DataFrame(data, columns=["trade_date", "high", "low", "close"])
    # print(df)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 将 trade_date 设置为索引并按索引排序
    df.set_index("trade_date", inplace=True)
    df.sort_index(inplace=True)

    # 计算每月的 ATR 总和
    monthly_atr_sum = calculate_monthly_atr_sum(df)

    # 绘制每月 ATR 总和
    plot_monthly_atr(monthly_atr_sum)

    # 正确关闭游标和连接
    cursor.close()
    conn.close()


# 绘制布林带图
def calculate_bollinger_bands(
    df, window=20
):  # 计算布林带各项指标，供plot_stock_bollinger_bands()调用
    df["sma"] = df["close"].rolling(window=window).mean()
    df["std_dev"] = df["close"].rolling(window=window).std()
    df["upper_band"] = df["sma"] + (2 * df["std_dev"])
    df["lower_band"] = df["sma"] - (2 * df["std_dev"])
    return df


def plot_bollinger_bands(df):  # 绘画布林带图，供plot_stock_bollinger_bands()调用
    plt.figure()
    plt.plot(df.index, df["close"], label="Close Price", color="blue")
    plt.plot(df.index, df["sma"], label="SMA", color="green")
    plt.plot(df.index, df["upper_band"], label="Upper Band", color="red")
    plt.plot(df.index, df["lower_band"], label="Lower Band", color="purple")
    plt.fill_between(
        df.index, df["upper_band"], df["lower_band"], color="grey", alpha=0.1
    )
    plt.title("Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    # plt.show()
    save_path = "static/images/Bollinger.png"
    plt.savefig(save_path, dpi=600)
    plt.close()


def plot_stock_bollinger_bands():  # 使用这个方法体画布林带
    conn = sqlite3.connect("Stock.db")
    cursor = conn.cursor()
    query = f"SELECT trade_date, open, high, low, close FROM Stock WHERE  trade_date >= '20230101'  ORDER BY trade_date ASC"
    data = cursor.execute(query).fetchall()
    cursor.close()
    conn.close()

    # if not data:
    #   print(f"No data found for stock code {stock_code}")
    #   return

    df = pd.DataFrame(data, columns=["trade_date", "open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df.sort_index(inplace=True)

    df = calculate_bollinger_bands(df)
    plot_bollinger_bands(df)


# 格式化日期
def format_date(date_str):
    """将日期字符串从 'YYYY-MM-DD' 格式转换为 'YYYY年M月D日' 格式"""
    date_object = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{date_object.year}年{date_object.month}月{date_object.day}号"


# 查询适合买入的时间
def query_golden_cross_dates():
    """查询Golden_cross表中的日期，只保留年月日"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT strftime('%Y-%m-%d', date) FROM Golden_cross")
        dates = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    # 提取日期列表
    return [format_date(date[0]) for date in dates if date[0] is not None]


# 查询适合抛出的时间
def query_death_cross_dates():
    """查询Death_cross表中的日期，只保留年月日"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT strftime('%Y-%m-%d', date) FROM Death_cross")
        dates = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    # 提取日期列表
    return [format_date(date[0]) for date in dates if date[0] is not None]


# 查询股票名字
def get_stock_info(ts_code):
    """根据股票代码查询Stock_List表中的信息"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    try:
        query = "SELECT * FROM Stock_List WHERE ts_code = ?"
        cursor.execute(query, (ts_code,))
        stock_info = cursor.fetchone()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

    if stock_info:
        # 将查询结果转换为字典，方便后续处理
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, stock_info))
    else:
        return None


def get_today_stock_info():
    """查询Stock表中今天日期的信息，并返回第一条记录的字典"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    # 获取今天的日期，格式化为YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")

    try:
        query = "SELECT * FROM Stock  ORDER BY trade_date DESC LIMIT 1"
        cursor.execute(query)
        stock_info_today = cursor.fetchone()  # 只获取第一行数据
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

    # 如果有查询结果，转换为字典
    if stock_info_today:
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, stock_info_today))
    else:
        return {}  # 如果没有数据，返回空字典


def query_pr_golden_cross_future():
    """查询PR_Golden_cross表中的Future值"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT Future FROM PR_Golden_cross")
        future_values = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    # 提取Future值列表
    return [value[0] for value in future_values if value[0] is not None]


def query_pr_death_cross_future():
    """查询PR_Death_cross表中的Future值"""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT Future FROM PR_Death_cross")
        future_values = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    # 提取Future值列表
    return [value[0] for value in future_values if value[0] is not None]


######################main################################
# Basic_information()  # 所有沪深股票公司的相关内部信息
# Get_Stock_List()  # 输入关键字去找相关股票
# Get_Stock()  # 输入股票代码获取该股票所有股市信息
# GetGD_x()  # 获取最近的投资，抛出时间，并输出图
# Get_Future()  # 预测后30股票日适合投资，抛出时间，并输出图
######################################################


app = Flask(__name__)


@app.route("/stock_detail", methods=["GET", "POST"])
def stock_detail():
    stock_detail_data = None
    if request.method == "POST":
        stock_detail_code = request.form.get("stock_detail_code", "")
        if stock_detail_code:
            stock_detail_data = Get_Stock(stock_detail_code)
            print("写入成功")
            GetGD_x()
            Get_Future()
    return render_template("stock.html", data=None, stock_detail=stock_detail_data)


# 确保 index 视图函数也传递 stock_detail 参数
@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    stock_detail_data = None
    if request.method == "POST":
        stock_code = request.form.get("stock_code", "")
        if stock_code:
            data = Get_Stock_List(stock_code)
    return render_template("stock.html", data=data, stock_detail=stock_detail_data)


@app.route("/stock_analysis", methods=["GET", "POST"])
def stock_analysis():
    stock_detail_data = None
    if request.method == "POST":
        stock_detail_code = request.form.get("stock_detail_code", "")
        basic_info = get_company_info(stock_detail_code)
        if stock_detail_code:
            Get_Stock(stock_detail_code)
            stock_detail_data = get_today_stock_info()
            GetGD_x()
            Get_Future()
            plot_volume_amount_and_pct_change()
            Get_ATR()
            plot_stock_bollinger_bands()
            stock_name = get_stock_info(stock_detail_code)["name"]
            basic_info=get_company_info(stock_detail_code)
            buy_date = query_golden_cross_dates()
            sell_date = query_death_cross_dates()
            future_death = query_pr_death_cross_future()
            future_golden = query_pr_golden_cross_future()
            k_suggest = GetDailyCandlestickChart()

    return render_template(
        "new_stock.html",
        data=None,
        stock_detail_data=stock_detail_data,
        basic_info= basic_info,
        stock_name=stock_name,
        buy_date=buy_date,
        sell_date=sell_date,
        k_suggest=k_suggest,
        future_death=future_death,
        future_golden=future_golden,
    )


if __name__ == "__main__":
    app.run(debug=True)

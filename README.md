<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" >

<title>沪深股票分析系统 | superzlf-evo</title>


 <h1 id="介绍">介绍</h1>
<p>我们推出了一款专为沪深股市量身定制的股票分析系统。它是一款精心打造的工具，为用户提供简洁透明的股市行情。它的主要目标是帮助用户理解股市的日常波动，同时提供相关股票的背景信息。此外，它简化了复杂的市场数据，提供直观的股价波动图表，使投资者能够清晰地了解日常市场动态。 通过无缝地集成先进的算法和实时数据，我们的股票分析系统不仅仅是一个数据展示工具的角色；它成为了一个强大的决策支持系统。用户可以轻松地访问关键指标，对个股进行深入的研究，和洞察整体市场趋势。系统的直观界面确保即使是对金融市场知识有限的用户也可以快速地浏览和参与股票交易。系统自动分析沪深股市的数据，准确地捕捉每只股票的每日涨跌，以易于消化和直观的方式呈现给用户。用户可以轻松地浏览每日涨跌，从而快速把握市场趋势。系统的设计以用户友好为出发点，迎合了那些不太熟悉股市的人，使他们能够轻松地理解和获取关键信息。</p>
<h1 id="获取数据">获取数据</h1>
<p>tushare网： https://www.tushare.pro/<br>
Tushare是一个免费、开源的python财经数据接口包。它可以为您提供沪深股市的实时行情数据、历史数据、财报数据、基金数据等，同时还提供了各种强大的数据分析和处理工具。<br>
您可以使用Tushare来获取和分析股票等金融数据，为您的投资决策提供便利和支持。您只需要安装Tushare库，注册一个账号，获取一个token，就可以开始使用Tushare的各种功能了。<br>
Tushare返回的绝大部分的数据格式都是pandas DataFrame类型，非常便于用pandas/NumPy/Matplotlib进行数据分析和可视化。您可以使用一些常用的pandas函数来操作和处理数据，例如to_x函数、iloc函数、sort_values函数、loc函数、mean函数、append函数等。</p>
<h1 id="存储数据">存储数据</h1>
<p>使用SQLITE存数据<br>
数据库设计方面数据库为Stock.db,库里有7个表：<br>
Stock表：实时获取的某个股票的2023至今的股市数据存进表里<br>
Stock_List表：存储沪深股票的一个简单信息<br>
Basic_infor表：存储所有沪深股票的对应公司的详细信息<br>
Death_cross表：存储过去100天内股票适合卖出的时间信息<br>
Golden_cross表：存储过去100天内股票适合买入的时间信息<br>
PR_Death_cross表：存储未来30股票交易日的死叉的时间节点数据<br>
PR_Golden_cross表：存储未来30股票交易日的金叉的时间节点数据</p>
<h1 id="功能实现以珠海某力电器为例">功能实现（以珠海某力电器为例）</h1>
<h2 id="k线图日线月线">K线图（日线，月线）</h2>
<p>使用SQL查询操作获取了股票今年内每天的开收盘价和最高最低价，并用此数据绘制k线图。<br>
这种图表的主要优点包括：<br>
信息丰富：除了价格变动趋势，它还显示了每个时间周期的开盘价、收盘价、最高价和最低价。<br>
情绪展示：通过对比开盘价和收盘价的相对位置，投资者可以迅速判断市场是看涨还是看跌。蜡烛图的颜色和长度帮助展示市场情绪。</p>
<h3 id="日线图">日线图 ：</h3>
<figure data-type="image" tabindex="1"><img src="https://superzlf-evo.github.io/post-images/1705215729201.png" alt="" loading="lazy"></figure>
<h3 id="月线图">月线图：</h3>
<figure data-type="image" tabindex="2"><img src="https://superzlf-evo.github.io/post-images/1705215740763.png" alt="" loading="lazy"></figure>
<h3 id="rsi指数与sma分析进行预测">RSI指数与SMA分析进行预测</h3>
<h2 id="用查询所得数据计算出rsi和sma指数来预测当前的股价走势通过rsi指数和sma指数输出当前针对股价的推荐操作并且输出给前端调用">用查询所得数据计算出RSI和SMA指数来预测当前的股价走势；通过RSI指数和SMA指数输出当前针对股价的推荐操作，并且输出给前端调用。<br>
<img src="https://superzlf-evo.github.io/post-images/1705215890828.png" alt="" loading="lazy"></h2>
<h2 id="过去适合买入卖出分析">过去适合买入卖出分析</h2>
<p>使用SQL查询操作获取Stock表中股票过去100天内的的收盘价close，和时间time。<br>
计算5日平均线和30日平均线，绘制线图并存储，供前端调用。<br>
5日平均线和30日平均线交点即为金叉或者死叉从而获取最适合买入卖出的时间节点并使用计算出数据更新<br>
Golden_cross表和Death_cross表。</p>
<figure data-type="image" tabindex="3"><img src="https://superzlf-evo.github.io/post-images/1705216050390.png" alt="" loading="lazy"></figure>
<hr>
<h2 id="未来30股票交易日适合买入卖出分析">未来30股票交易日适合买入卖出分析</h2>
<p>使用SQL查询操作获取Stock表中股票过去100天内的的收盘价close，和时间time。<br>
使用ARIMA模型预测未来30股票交易日的该股票收盘价。<br>
计算5日平均线和30日平均线，画图。获取未来最适合买入卖出的时间节点并使用计算出数据更新PR_Golden_cross表，PR_Death_cross表。</p>
<pre><code class="language-python"># 定义ARIMA模型并训练模型
    model = ARIMA(data, order=(3, 2, 0))  # 参数配置，如p、d、q等。
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(data), end=len(data) + 30)  # 预测未来30天的收盘价。 
</code></pre>
<figure data-type="image" tabindex="4"><img src="https://superzlf-evo.github.io/post-images/1705216257392.png" alt="" loading="lazy"></figure>
<hr>
<h2 id="atr分析">ATR分析</h2>
<p>使用SQL查询操作获取股票今年内的数据：<br>
1.计算周期内的最高价和最低价；<br>
2.计算收盘价与最高价和最低价之间的最大值和最小值；<br>
3.计算真实范围（即最大值与最小值的差值）；<br>
4.计算真实范围的平均值得到ATR指标。并绘制了一个柱状图显示每月的ATR总和，展示了股票在每个月内价格波动的总体趋势。<br>
<img src="https://superzlf-evo.github.io/post-images/1705216355319.png" alt="" loading="lazy"></p>
<hr>
<h2 id="布林带分析">布林带分析</h2>
<p>使用SQL查询操作获取股票今年内的数据，计算出布林带指标：<br>
1.计算移动平均线（中轨线）：将周期的收盘价相加再除以该周期的天数。<br>
2.计算标准差：先计算每个交易日收盘价与移动平均线之间的差值，然后取这些差值的平方并求和，最后除以周期天数，并开平方根得到标准差。<br>
3.计算上轨线和下轨线：上轨线等于移动平均线加上标准差的乘积，下轨线等于移动平均线减去标准差的乘积.<br>
<img src="https://superzlf-evo.github.io/post-images/1705216467937.png" alt="" loading="lazy"></p>
<hr>
<h2 id="界面设计">界面设计</h2>
<p>前端设计主要分为index和stock页面，index页面用户通过输入股票关键字来查询相关的股票代码，接着根据股票代码查询该股票的具体信息。stock页面则对股票进行具体的分析，配有图文展示。后端采用的是Flask框架，通过request中的post请求与前端进行交互。</p>
<figure data-type="image" tabindex="5"><img src="https://superzlf-evo.github.io/post-images/1705216655931.png" alt="" loading="lazy"></figure>
<figure data-type="image" tabindex="6"><img src="https://superzlf-evo.github.io/post-images/1705216664832.png" alt="" loading="lazy"></figure>
<hr>




  
    <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>
 




  </body>
</html>

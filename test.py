import yfinance as yf

news = yf.Search("Google", news_count=10).news
# print(news)
# Create a Ticker instance for Apple
aapl = yf.Ticker("AAPL")

# Get earnings history using get_earnings_history() method
earnings_history = aapl.get_earnings_history()
news = aapl.news
print("\nNews for AAPL:")
print(news)
# Print the earnings history
print("\nEarnings History for AAPL:")
print(earnings_history)

# You can also get earnings history for other stocks
msft = yf.Ticker("MSFT")
print("\nEarnings History for MSFT:")
print(msft.get_earnings_history())


# news = yf.Search("Google", news_count=10).news
#news = yf.earnings_estimate("Google", news_count=10).news
# data = yf.Ticker.eps_trend()
# print(data)
# url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=2d&modules=assetProfile"
# url = f"https://query1.finance.yahoo.com/v8/finance/quoteSummary/AAPL?modules=earnings"
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#     'Accept': 'application/json',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Accept-Language': 'en-US,en;q=0.9',
# }

# response = requests.get(url, headers=headers)
# print(response)
# if response.status_code == 200:
#     data = response.json()
#     print(data)                    

# # Fetch earnings data
# earnings_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=earnings"
# earnings_response = requests.get(earnings_url)
# earnings_data = earnings_response.json()
# print(earnings_data)

# # Fetch latest news
# news_url = "https://query1.finance.yahoo.com/v7/finance/news?lang=en-US&region=US&symbols=AAPL"
# news_response = requests.get(news_url)
# news_data = news_response.json()
# print(news_data)


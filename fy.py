import yfinance as yf
news = yf.Search("Google", news_count=10).news
# Extract and print only the titles
for i, item in enumerate(news, 1):
    print(f"{i}. {item['title']}")

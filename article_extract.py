import urllib
import random
import requests
from time import sleep, time
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
from stringcase import snakecase

MEDIUM_BASE_URL = "https://medium.com/topic/%s"

CATEGORIES = {
    'all': [
        "sports", "books", "TV", "space", "mental-health", "pets", "parenting", "psychology",
        "sexuality", "travel", "cities", "history", "environment", "food", "beauty"
    ]

    # 'industry': [
    #     "business", "marketing", "economy", "venture-capital"
    # ],
    #
    # 'arts': [
    #         "book", "fiction", "poetry", "comics",
    #         "writing"
    # ],
    #
    # 'innovation': [
    #     "blockchain", "data-science", "cybersecurity", "machine-learning",
    #     "programming"
    # ],
    #
    # 'life': [
    #     "health", "lifestyle", "mental-health", "fitness",
    #     "psychology"
    # ]
}
#
# # Shuffle the categories to make sure we are not exhaustively crawling only the first categories
categories = list(CATEGORIES.items())
# random.shuffle(categories)

for category_name, keywords in categories:
    print("Exploring Category=\"{0}\"".format(category_name))
    for kw in keywords:
        # Get trending content from Pocket's explore endpoint
        #
        result = requests.get(MEDIUM_BASE_URL % urllib.parse.quote_plus(kw))
        # Extract the media items
        soup = BeautifulSoup(result.content, "html5lib")
        media_items = soup.select("h2 + div > section > div > section > div > div > div > h3 > a[href]")
        for item in media_items:
            if item['href'].startswith('https'):
                url = item['href']
            else:
                url = ('https://medium.com'+item['href'])
            print(url)
            try:
                article = Article(url)
                article.download()
                article.parse()
                content = article.text
            except ArticleException as e:
                print("Encoutered exception when parsing \"{0}\": \"{1}\"".format(url, str(e)))
                continue

            if not content:
                print("Couldn't extract content from \"{0}\"".format(url))
                continue
            # Save the text file
            file_name = "{0}.txt".format(str(snakecase(item.text.replace("/", ''))))
            with open('./data/categories/{1}/{0}'.format(file_name, category_name), 'w+') as text_file:
                text_file.write(content)
            # Need to sleep in order to not get blocked
            sleep(random.randint(5, 15))

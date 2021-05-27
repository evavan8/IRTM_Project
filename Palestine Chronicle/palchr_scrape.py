import requests
from bs4 import BeautifulSoup
import pandas as pd

""" Purpose of this script 
This script stores for each news item on 'https://www.palestinechronicle.com/category/articles/' (up to a specified page)
the news headline, date, link to the full article, and text of the full article in a csv file named 
'PalestineChronicle_News.csv'. 
Number of results on May 24 2021: 5971 (much more articles in later years -- take into account when processing)
"""

print_progress = True
scrape_headlines = True
scrape_bodies = True
continue_processing = True  # continue where we left off last time (don't start over completely)

#--------------------- Scrape headlines, dates, links to full articles -------------------

if scrape_headlines:
    last_page = 200     #Goes back to Nov 8, 2010

    page_core = 'https://www.palestinechronicle.com/category/articles/page/'    # without page number

    palchr_list = []  # store 'title' (headline) - 'date' (date) - 'link' (link to full news article)

    for curr_page_no in range(1, last_page):
        if print_progress: print(curr_page_no)
        curr_page = page_core + str(curr_page_no)
        page = requests.get(curr_page)
        soup = BeautifulSoup(page.content, 'html.parser')

        headers = soup.find_all('header', class_='mh-posts-list-header')

        for h in headers:
            link = h.find('h3', class_='entry-title').find('a')['href']
            if not link.startswith('https'):
                continue
            date = h.find('div', class_='mh-meta').find('a').text
            title = h.find('h3', class_='entry-title').find('a').text
            strip_chars = ['\r', '\n', '\t']
            for sc in strip_chars:
                title = title.strip(sc)
            palchr_list.append([title, date, link])

    df = pd.DataFrame.from_records(palchr_list, columns=['headline', 'date', 'link'])
    df.to_csv('PalestineChronicle_News.csv', index=False)

#--------------------- Scrape full text of article from link stored -------------------

if scrape_bodies:
    df = pd.read_csv('PalestineChronicle_News.csv')

    """ test links
    link_plain = 'https://www.palestinechronicle.com/sa-rabbi-denies-existence-of-apartheid-in-israel/'
    link_tweets = 'https://www.palestinechronicle.com/mahrez-waves-palestinian-flag-to-celebrate-manchester-citys-win-premier-league-title-video/'
    link_pictures = 'https://www.palestinechronicle.com/voices-from-gaza-two-mothers-speak-to-the-palestine-chronicle-from-the-shifa-hospital-photos/'
    link_quotes = 'https://www.palestinechronicle.com/freepalestine-international-athletes-speak-out-for-palestine/'
    """

    # add new column to df
    df['body'] = ['' for _ in range(len(df))]


    for i in range(len(df)):
        if print_progress: print(i)
        link = df['link'][i]
        if continue_processing:
            if df['body'][i] != '': continue  # continuing processing from where we left off last time
        page = requests.get(link)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            article = soup.find('div', class_='entry-content')
            all_p = article.find_all('p')
            body = []
            for p in all_p:
                body.append(p.get_text())
            body = ' '.join(body)
            df['body'][i] = body
        if i % 100 == 0:    # flush to disk every 100 articles
            df.to_csv('PalestineChronicle_News.csv', index=False)


import requests
from bs4 import BeautifulSoup
import pandas as pd

""" Purpose of this script 
This script stores for each news item on 'https://english.wafa.ps/Regions/Details/2?pageNumber=1' (as far as the history
goes back) the news headline, date & time, link to the full article, and text of the full article in a csv file named 
'Wafa_News_Occupation.csv'. 
Number of results on May 19 2021: 1347
"""


last_page = False
curr_page = 'https://english.wafa.ps/Regions/Details/2?pageNumber=1'
wafa_list = []  # store 'title' (headline) - 'date-time' (date & time) -
                #       'link' (link to full news article) - 'txt' (text of full news article)

print_progress = True

while(not last_page):   # while we have not reached the last page
    page = requests.get(curr_page)
    soup = BeautifulSoup(page.content, 'html.parser')

    # store all headlines with dates & links to actual news article (in wafa_list)
    rows = soup.find_all('div', class_='content')
    for r in rows:
        if r.find('h4', class_='glyphicon') is None and r.find('h4', class_='title') is not None:    # don't want titles appearing on the side of the page - have class glyphicon
            # get date & time, title, and link to full news article
            date_time = r.find('span', class_='date').text
            title = r.find('h4', class_='title').text
            link = 'https://english.wafa.ps' + r.find('a')['href']

            # retrieve text of full news article
            page2 = requests.get(link)
            if page2.status_code == 200:
                soup2 = BeautifulSoup(page2.content, 'html.parser')
                try:
                    txt = soup2.find('div', class_='content').text
                except:
                    txt = ''
            else:
                txt = 'PAGE LOAD ERROR'

            # store everything in wafa_list
            wafa_list.append((title, date_time, link, txt))

    # navigate to the next page
    curr_page_no = soup.find('li', class_='active-page').text.split('\n')[1]
    if print_progress: print(f'Page number: {curr_page_no}')
    prev_next_buttons = soup.find_all('a', class_='btn')

    if len(prev_next_buttons) == 2:     # check if there is still a 'next' page
        curr_page = 'https://english.wafa.ps' + soup.find_all('a', class_='btn')[1]['href']
    else:
        print(f'REACHED LAST PAGE: {curr_page_no}')
        last_page = True


df = pd.DataFrame.from_records(wafa_list, columns=['headline', 'date_time', 'link', 'full_text'])
df.to_csv('Wafa_News_Occupation.csv', index=False)



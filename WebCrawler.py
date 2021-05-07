#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the libraries

from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
from urllib.request import urlparse
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time
import re
import os

#docs_directory = "uic-docs-text/"

class webCrawler:
    
    def __init__(self, root_url):
        self.root_url = root_url
        self.pool = ThreadPoolExecutor(max_workers = 10)
        self.allowed_domain = 'uic.edu'
        self.invalid_extensions = (".pdf", ".jpg", ".jpeg", ".doc", ".docx", ".ppt", ".pptx", ".png", ".txt", ".exe", ".ps", ".psb")
        #Keep track of the visited pages
        self.visited_pages = set([])
        self.queue_to_crawl = Queue()
        self.counter = 0
        
        #Adding the root_url to the queue
        self.queue_to_crawl.put(self.root_url)
    
    def is_valid_extension(self, url):
        if url.lower().endswith(self.invalid_extensions):
            return False
        return True
    
    def parse(self, res, root_url):
        temp_urls = []
        domain = 'uic.edu'
        prefixes = ('http', 'https')
        internal_links = set()
        external_links = set()
        beautiful_soup_object = BeautifulSoup(res.content,"lxml")
        for anchor in beautiful_soup_object.findAll("a"):
            href = anchor.attrs.get("href")
            if(href != "" or href != None):
                href = urljoin(root_url, href)
                href_parsed = urlparse(href)
                href = href_parsed.scheme
                href += "://"
                href += href_parsed.netloc
                href += href_parsed.path
                final_parsed_href = urlparse(href)
                is_valid = bool(final_parsed_href.scheme) and bool(final_parsed_href.netloc)
                if is_valid:
                    if((domain not in href and href not in external_links)):
                        external_links.add(href)
                    if((domain in href and href not in internal_links and href.startswith(prefixes))):
                        internal_links.add(href)
                        temp_urls.append(href)
                        self.queue_to_crawl.put(href)
    
    def document_creation(self, content, filename, url = None):
        with open(filename, 'w', encoding='utf-8', errors = 'ignore') as f:
            if url is not None:
                f.writelines("<URL>" + url + "</URL>" + "\n")
            f.write(str(content))
    
    def scraping_info(self, content, url, counter):
        save_path = "C:/Users/apoor/Downloads/IR_Assignments/Course_Project/uic-docs-text/"
        self.document_creation (content, os.path.join(save_path + str(counter) + ".txt"), url)
        
    def download_page(self, url):
        r = requests.get(url)
        res = r.text
        soup = BeautifulSoup (res, 'html.parser')
        
        for script in soup.findAll('script'):
            script.decompose()
        for style in soup.findAll('style'):
            style.decompose()
        
        texts = soup.get_text()
        
        #checking if the page can be downloaded
        if r.status_code == 200:
            self.counter += 1
            self.parse(r, url)
            self.scraping_info(texts, url, self.counter)
 
    def run_scraper(self):
        while self.counter <= 3000:
            try:
                #Popping the URL from the front of the queue
                current_url = self.queue_to_crawl.get()
                #Check if the URL has a valid extension
                bool_is_valid_extension = self.is_valid_extension(current_url)
                #Removing the slash from the end of URL to avoid duplicate URL
                current_url = current_url.strip("/")
                #Scrape the URLS only which are not there in the visited pages list
                if current_url not in self.visited_pages and bool_is_valid_extension:
                    self.visited_pages.add(current_url)
                    #Downloading the page for the current_url
                    #Implementing the multi threading in order to download many pages to save time
                    
                    job = self.pool.submit(self.download_page ,current_url)
                    #self.download_page(current_url)
            except Exception as e:
                #print(e)
                continue
                    
                

if __name__ == '__main__':
    start_time = time.time()
    r = webCrawler("https://www.cs.uic.edu/")
    r.run_scraper()
    end_time = time.time()
    #print("the program execution time is ===",end_time - start_time)


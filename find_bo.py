import os
import logging
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from curl_cffi import requests
from openai import OpenAI
import tiktoken
import html_text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, 
    filename='find_bo.log', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# TODO: Use the appropriate api endpoint (maybe, parse or something)
def find_business_owner(page_text):
    prompt = f"""
    Extract ONLY the business owner's name from this text.

    Rules:
    - Return ONLY the name, nothing else
    - If no owner name is found, return exactly: 'none'
    - Do not return employee names, CEO names unless explicitly stated as owner

    Text: {page_text}
    """
    response = client.responses.create(
        model='gpt-4o-mini',
        input=prompt,
        temperature=0.0
    )
    result = response.output_text.strip()
    if result.lower() in ['', 'none', 'no owner', 'no name', 'not found', 'n/a', 'not mentioned']:
        return None
    return result

def truncate_text(text):
    MAX_TOKENS = 2000
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    print(f"Original token count: {len(tokens)}")
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = encoding.decode(tokens)
    return text

def internal_page_to_search(all_pages_links):
    prompt = f"""
    You are an expert in finding business owners' names from website content. Given a list of internal links from a website, choose one link that is most likely to contain the business owner's name.
    
    Internal Links: {', '.join(all_pages_links)}
    
    Output Format: the chosen internal link unmodified
    """
    response = client.responses.create(
        model='gpt-4o-mini',
        input=prompt,
    )
    return response.output_text.strip()

def generate_analytics(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    all_pages = get_all_internal_links(response.url, soup)

    encoding = tiktoken.get_encoding("cl100k_base")
    text = html_text.extract_text(response.text, guess_layout=False)
    tokens = encoding.encode(text)
    return {
        'url': response.url,
        'token_count': len(tokens),
        'links': all_pages,
    }

def is_internal_link(base_url, link):
    """Checks if the link is an internal link of the base_url."""
    parsed_base = urlparse(base_url)
    parsed_link = urlparse(link)
    # An internal link either has the same domain or is a relative link
    return (parsed_link.netloc == '' or parsed_link.netloc == parsed_base.netloc)

def get_all_internal_links(base_url, soup):
    """Finds all internal links within the website homepage, excluding 'tel:' and 'mailto:' links."""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        # Exclude 'tel:' and 'mailto:' links
        if link.startswith(('tel:', 'mailto:', 'javascript:')) or re.search(r'\.(png|jpg|jpeg|pdf)$', link):
            continue 
        try:
            full_url = urljoin(base_url, link)
        except ValueError:  # I got an invalid internal URL even after the above filtering
            logging.error(f"Error joining URL {base_url} and link {link}")
            continue
        if is_internal_link(base_url, full_url):
            links.add(full_url)
    return list(links)[:20]

def get_about_link(all_page_links):
    for link in all_page_links:
        parsed_link = urlparse(link)
        if 'about' in parsed_link.path.lower():
            return link
    return None

def make_request(url):
    try:
        response = requests.get(url, impersonate='chrome')
    except requests.RequestsError as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    if response.status_code != 200:
        logging.error(f"invalid response code {url}, status code: {response.status_code}")
        return None
    return response


if __name__ == "__main__":
    start_time = datetime.now()
    urls_df = pd.read_csv('websites.csv').dropna(subset=['urls'])
    urls = urls_df['urls'].tolist()
    # urls = ['https://www.beautymarkbynina.com/']
    # websites_analytics = []
    bo_names = []
    for url in urls:
        response = make_request(url)
        if response is None:
            continue
        homepage_text = html_text.extract_text(response.text, guess_layout=False)
        all_pages_links = get_all_internal_links(url, BeautifulSoup(response.text, 'html.parser'))
        about_link = get_about_link(all_pages_links)

        about_page_text = ''
        if about_link:
            about_page_response = make_request(about_link)
            if about_page_response:
                about_page_text = html_text.extract_text(about_page_response.text, guess_layout=False)
            
        combined_website_text = homepage_text + '\n' + about_page_text
        text = truncate_text(combined_website_text)

        business_owner = find_business_owner(text)
        print(f"Business Owner for {url}: {business_owner}")
        bo_names.append({
            'url': url,
            'business_owner': business_owner
        })     
    df = pd.DataFrame(bo_names)
    df.to_csv('business_owners.csv', index=False)
    print(f"elapsed time: {datetime.now() - start_time}")

        # if business_owner == 'None':
        #    all_pages_links = get_all_internal_links(url, BeautifulSoup(response.text, 'html.parser'))
        #    if all_pages_links:
        #     internal_page = internal_page_to_search(all_pages_links)
        #     print(f"Chosen internal page to search: {internal_page}")
        










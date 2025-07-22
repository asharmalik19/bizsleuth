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
    - Look for titles like: owner, founder, director, creative director, president, principal, proprietor
    - Also consider someone who founded the business or is described as starting/establishing it
    - Return ONLY the name, nothing else
    - If multiple potential owners, return the most senior/primary one
    - If no owner name is found, return exactly: 'none'
    - Do not return regular employees, instructors, or staff unless they have ownership/leadership titles

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
    MAX_TOKENS = 3000
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

def get_relevant_links(all_page_links, pages_to_grab):
    """In addition to the about page link, grabs 1 other link from a list of potential
    pages from top to down priority"""
    relevant_pages_links = set()
    for link in all_page_links:
        parsed_link = urlparse(link)
        if 'about' in parsed_link.path.lower():
            relevant_pages_links.add(link)
        for page in pages_to_grab:
            if page in parsed_link.path.lower():
                relevant_pages_links.add(link)
                break
    return list(relevant_pages_links)

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
    PAGES_TO_GRAB = [
        'owner', 'founder', 'director', 'ceo', 'team', 'staff', 'instructor', 
        'trainer', 'coach', 'teacher', 'stylist', 'artist', 'therapist', 
        'esthetician', 'specialist', 'practitioner', 'doctor', 'members',
        'crew', 'meet', 'bio', 'story'
    ]
    start_time = datetime.now()
    urls_df = pd.read_csv('websites.csv').dropna(subset=['urls'])
    urls = urls_df['urls'].tolist()
    # urls = ['http://www.covingtonregionalballet.com/']
    # websites_analytics = []
    bo_names = []
    for url in urls:
        response = make_request(url)
        if response is None:
            continue
        website_text = html_text.extract_text(response.text, guess_layout=False)
        all_pages_links = get_all_internal_links(url, BeautifulSoup(response.text, 'html.parser'))     
        relevant_links = get_relevant_links(all_pages_links, PAGES_TO_GRAB)

        for link in relevant_links:
            relevant_page_response = make_request(link)
            if relevant_page_response is None:
                continue
            page_text = html_text.extract_text(relevant_page_response.text, guess_layout=False)
            website_text = website_text + '\n' + page_text      

        text = truncate_text(website_text)
        business_owner = find_business_owner(text)
        print(f"Business Owner for {url}: {business_owner}")
        bo_names.append({
            'url': url,
            'business_owner': business_owner
        })     
    df = pd.DataFrame(bo_names)
    df.to_csv('business_owners.csv', index=False)
    print(f"elapsed time: {datetime.now() - start_time}")








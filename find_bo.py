import os
import logging
import re
from datetime import datetime
from pydantic import BaseModel

import pandas as pd
from dotenv import load_dotenv
from curl_cffi import requests
from openai import OpenAI
import tiktoken
import html_text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.ERROR, 
    filename='find_bo.log', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class template(BaseModel):
    business_owner: str
    address: str
    summary: str
    business_size: str

# TODO: Use the appropriate api endpoint (maybe, parse or something)
def find_business_info(page_text):
    prompt = f"""
    Extract ONLY the following information from the text below:

    1. The business owner's name.
    2. The full business address.
    3. A brief summary of the business (no more than 1 sentence).
    4. The business size ("small", "medium", or "large").

    Rules:
    - For the owner, look for titles like: owner, founder, director, creative director, president, principal, proprietor.
    - Also consider someone who founded or established the business.
    - Return ONLY the name for the owner. If multiple, return the most senior/primary one. If none found, return exactly: 'none'.
    - For the address, return the full address as it appears in the text. If not found, return exactly: 'none'.
    - For the summary, provide a concise overview of what the business does or offers based on the text. If not enough information, return exactly: 'none'.
    - For the business size, categorize as:
        * "small" if the text explicitly says "small" or implies fewer than ~50 employees,
        * "medium" if it says "medium" or implies ~50–250 employees,
        * "large" if it says "large" or implies more than ~250 employees,
        * otherwise return exactly: 'none'.
    - Do not return any extra text, explanations, or unrelated information.
    - Format your answer as JSON with keys: "owner", "address", "summary", "business_size".

    Text: {page_text}
    """
    response = client.responses.parse(
        model='gpt-4o-mini',
        input=prompt,
        temperature=0.0,
        text_format=template
    )
    info = response.output_parsed
    return info

def truncate_text(text):
    MAX_TOKENS = 3000
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    print(f"Original token count: {len(tokens)}")
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = encoding.decode(tokens)
    return text

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
    return list(links)

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
        response = requests.get(
            url, 
            impersonate='chrome',
            timeout=30 
        )
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
    business_info_list = []
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
        business_info = find_business_info(text)
        print(f'info for {url}: {business_info}')
        business_info_list.append({
            'url': url,
            'business_owner': business_info.business_owner,
            'address': business_info.address,
            'summary': business_info.summary,
            'business_size': business_info.business_size
        })     
    df = pd.DataFrame(business_info_list)
    df.to_csv('business_info.csv', index=False)
    print(f"elapsed time: {datetime.now() - start_time}")








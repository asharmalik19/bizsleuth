import asyncio
import os
import logging
from datetime import datetime
from pydantic import BaseModel
import json

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken
import html_text
from crawl4ai import AsyncWebCrawler, BFSDeepCrawlStrategy
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# TODO: crawl more than 1 level deep
# TODO: make the code llm-agnostic
# TODO: implement proper batching or async processing for large-scale URLs
# TODO: use AI batch API

logging.basicConfig(
    level=logging.INFO,
    filename="bizsleuth_test.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

with open("long_prompt.txt", "r", encoding="utf-8") as file:
    PROMPT_TEMPLATE = file.read()


class BusinessData(BaseModel):
    estimated_revenue: int
    number_of_locations: int
    business_type: str
    contact_name: str
    contact_email: str
    number_of_members: int
    current_software: str
    services_offered: str
    ai_summary: str
    confidence_score: int
    street: str
    city: str
    state: str
    postal_code: str
    country: str
    company_name: str
    phone_number: str


class Email(BaseModel):
    subject_line: str
    greeting: str
    intro_body: str
    bullet_transition_sentence: str
    benefit_bullets: list[str]
    cta: str
    signature: str


class Emails(BaseModel):
    email_1: Email
    email_2: Email


class Template(BaseModel):
    business_id: str
    business_data: BusinessData
    emails: Emails


async def find_business_info(website_text, external_links_str, business_id) -> Template:
    prompt = f"""
    {PROMPT_TEMPLATE}
    Exteral links found on the website: {external_links_str}
    Website Text: {website_text}
    """
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=prompt,
        text_format=Template,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["vs_68c3379096548191839b3bdfa12b77a8"],
            }
        ],
    )
    info = response.output_parsed
    info.business_id = business_id
    return info


def truncate_text(text):
    MAX_TOKENS = 50000
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # TODO: tiktoken.get_encoding(MODEL)
    tokens = encoding.encode(text)
    print(f"Original token count: {len(tokens)}")
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = encoding.decode(tokens)
    return text


async def crawl_websites(urls_with_business_ids):
    browser_config = BrowserConfig(browser_type="chromium", headless=True) 
    run_config = CrawlerRunConfig(
        wait_until="networkidle",
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=1,
            max_pages=50,
            include_external=False
        )
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawled_results = []
        for _, row in urls_with_business_ids.iterrows():
            business_id = row['business_id']
            url = row['url']
            total_website_text = ""
            total_external_links = set()
            results = await crawler.arun(
                url=url,
                config=run_config
            )
            if not results:
                continue
            for result in results:
                if not result.success:
                    continue
                text = html_text.extract_text(result.html, guess_layout=False)
                total_website_text += text + "\n"
                external_links = result.links["external"]
                external_links_hrefs = [link["href"] for link in external_links]
                total_external_links.update(external_links_hrefs)
            total_website_text = truncate_text(total_website_text)
            total_external_links_str = ", ".join(total_external_links)
            crawled_results.append({
                "total_website_text": total_website_text,
                "total_external_links_str": total_external_links_str,
                "business_id": business_id,
            })
    return crawled_results


async def main(urls_with_business_ids):
    crawled_results = await crawl_websites(urls_with_business_ids)
    print("Finished crawling websites! Now running ai tasks...")
    ai_tasks = [
        find_business_info(
            result["total_website_text"],
            result["total_external_links_str"],
            result["business_id"],
        )
        for result in crawled_results
    ]
    ai_results = await asyncio.gather(*ai_tasks, return_exceptions=True)
    valid_ai_results = []
    for result in ai_results:
        if isinstance(result, Exception):
            logging.warning(f"AI task failed: {result}")
            continue
        valid_ai_results.append(result)
    business_info_list = [
        business_info.model_dump() for business_info in valid_ai_results
    ]
    return business_info_list


if __name__ == "__main__":
    start_time = datetime.now()
    BATCH_SIZE = 100
    business_info_total = []
    urls_with_business_ids = pd.read_csv("websites.csv").dropna(
        subset=["url"]
    )
    for batch in range(0, len(urls_with_business_ids), BATCH_SIZE):
        print(f"processing batch {batch} to {batch + BATCH_SIZE}")
        batched_df = urls_with_business_ids.iloc[batch : batch + BATCH_SIZE]
        batched_business_info = asyncio.run(main(batched_df))
        business_info_total.extend(batched_business_info)
        print(batched_business_info)
    with open("business_info.json", "w", encoding="utf-8") as f:
        json.dump(business_info_total, f, ensure_ascii=False, indent=2)

    logging.info(f"Total elapsed time: {datetime.now() - start_time}")

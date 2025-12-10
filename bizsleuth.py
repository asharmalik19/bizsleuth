import asyncio
import os
import logging
from datetime import datetime
from pydantic import BaseModel

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
    filename="bizlogger.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

with open("prompt.txt", "r", encoding="utf-8") as file:
    prompt = file.read()


class BusinessData(BaseModel):
    business_name: str
    business_owner_name: str
    business_address: str
    business_size: str
    contact_email: str
    phone_number: str
    number_of_locations: str
    business_summary: str
    url: str = ""  # Will be set after parsing


async def find_business_info(website_text, url) -> BusinessData:
    PROMPT_TEMPLATE = f"""
    {prompt}
    Website Text: {website_text}
    """
    response = await client.responses.parse(
        model="gpt-4o-mini",
        input=PROMPT_TEMPLATE,
        text_format=BusinessData
    )
    info = response.output_parsed
    info.url = url
    return info


def truncate_text(text):
    MAX_TOKENS = 20000
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # TODO: tiktoken.get_encoding(MODEL)
    tokens = encoding.encode(text)
    print(f"Original token count: {len(tokens)}")
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = encoding.decode(tokens)
    return text


async def crawl_websites(urls_list):
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
        for url in urls_list:
            total_website_text = ""
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
            total_website_text = truncate_text(total_website_text)
            crawled_results.append({
                "url": url,
                "website_text": total_website_text,
            })
    return crawled_results


async def main(urls_list):
    crawled_results = await crawl_websites(urls_list)
    print("Finished crawling websites! Now running ai tasks...")
    ai_tasks = [
        find_business_info(
            result["website_text"],
            result["url"],
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
    urls_df = pd.read_csv("websites.csv").dropna(
        subset=["url"]
    )
    urls = urls_df['url'].tolist()
    for batch in range(0, len(urls), BATCH_SIZE):
        print(f"processing batch {batch} to {batch + BATCH_SIZE}")
        batched_urls = urls[batch : batch + BATCH_SIZE]
        batched_business_info = asyncio.run(main(batched_urls))
        business_info_total.extend(batched_business_info)
        print(batched_business_info)
    output_df = pd.DataFrame(business_info_total)
    output_df.to_csv("business_info.csv", index=False, encoding="utf-8")

    logging.info(f"Total elapsed time: {datetime.now() - start_time}")

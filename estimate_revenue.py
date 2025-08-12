import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class Revenue(BaseModel):
    monthly_revenue_usd: Optional[int] = None
    monthly_revenue_usd_range: Optional[List[int]] = None

def get_monthly_revenue(website):
    prompt = f"""Role: You are a precise web researcher + revenue estimator. Use only online evidence. Do not invent facts.

Input:
{website}

Task: Estimate the business’s monthly revenue in USD.

Rules (strict):

If the brand operates multiple physical locations, return the company-wide total monthly revenue (not a single store).

If it’s a single location (or no clear multi-location evidence), return that entity’s monthly revenue.

You don’t know the address. Infer locality (city/state/country) from the website footer/contact/about pages, or from credible listings (Google Business Profile, Yelp, press). If nothing reliable is found, proceed with national pricing/benchmarks and say assumptions in your head but do not output them.

Favor official pricing pages. Use memberships/subscriptions, packs, or per-service rates. If none, use credible third-party signals (interviews, filings, reputable news/reports).

Use defensible proxies for demand (review counts, class schedules × capacity × load factor, attendance figures, traffic/footfall where available).

Prefer low/base/high internally, but output only a single number OR a range.

If evidence is truly insufficient, output null (not a string).

Computation guidance (choose what fits the business):

Membership model: members × monthly fee (+ drop-ins/packs if applicable).

Attendance × yield: monthly attendances × effective revenue per visit (below rack rate if memberships dominate).

Services model: avg jobs/month × avg ticket.

Retail/e-com: monthly orders × AOV (if the site is clearly commerce-focused).

Exclude non-attributable national digital products unless clearly part of the same entity’s revenue.

Output format (MUST be valid JSON and NOTHING else):

If a single best estimate:
    {{"monthly_revenue_usd": 0}}
If a defensible range:
    {{"monthly_revenue_usd_range": [0, 0]}}
If not enough data:
    {{"monthly_revenue_usd": null}}

Quality bar: Use at least one primary/official source when possible. Prefer conservative assumptions over speculative leaps.
"""

    response = client.responses.parse(
        model='gpt-4.1',
        input=prompt,
        temperature=0.0,
        text_format=Revenue,
        tools=[{"type": "web_search_preview"}]
    )
    return response.output_parsed


if __name__=='__main__':
    website = 'https://www.yogavida.com/'
    est_revenue = get_monthly_revenue(website)
    print(est_revenue.model_dump_json(indent=2))
    

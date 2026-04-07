#!/usr/bin/env python3
"""
Fetch latest IEEPA tariff refund news using Claude API with web search,
then write structured results to news-data.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

NEWS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news-data.json")

SYSTEM_PROMPT = """You are a trade policy research assistant. Search the web for the latest news about IEEPA tariff refunds in the United States.

Focus on:
- CBP CAPE system updates and launch status
- Court of International Trade (CIT) rulings related to IEEPA tariff refunds
- Supreme Court follow-up rulings or government appeals
- CBP bulletins (CSMS) about refund processing
- New developments in the tariff refund process
- Changes to ACH enrollment requirements
- Section 122 replacement tariff updates
- Any new class action lawsuits related to tariff refunds
- Industry analysis from law firms (Crowell & Moring, Thompson Coburn, etc.)

Return your findings as a JSON array of news items. Each item should have:
- "date": the date of the news (e.g., "APRIL 2, 2026")
- "badge": a short label (e.g., "BREAKING", "UPDATE", "NEW DETAIL", "WATCH LIST")
- "category": one of "breaking", "update", "detail", "positive", "legal", "watch"
- "headline": a concise headline (under 80 chars)
- "summary": a 2-4 sentence summary of the development and what it means for importers
- "source": the source publication or organization

Return ONLY valid JSON — an array of objects. No markdown, no code fences. Return between 3 and 8 news items, ordered by importance/recency. If you cannot find recent news, return fewer items about the most recent developments you can find."""

USER_PROMPT = """Search for the latest news and developments about IEEPA tariff refunds, the CBP CAPE system, and related Court of International Trade rulings. Focus on developments from the past week. Today's date is """ + datetime.now(timezone.utc).strftime("%B %d, %Y") + "."


def call_claude_api():
    """Call the Anthropic API with web search tool enabled."""
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10
            }
        ],
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": USER_PROMPT}
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"API error {e.code}: {body}")
        sys.exit(1)


def extract_text_from_response(response):
    """Extract all text blocks from the Claude response."""
    texts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            texts.append(block["text"])
    return "\n".join(texts)


def parse_news_items(text):
    """Parse JSON news items from Claude's response text."""
    # Try to find JSON array in the text
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        print("Warning: Could not find JSON array in response")
        print("Response text:", text[:500])
        return []

    json_str = text[start:end + 1]
    try:
        items = json.loads(json_str)
        if not isinstance(items, list):
            print("Warning: Parsed JSON is not a list")
            return []
        return items
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        print("JSON string:", json_str[:500])
        return []


def strip_citations(text):
    """Remove <cite ...>...</cite> tags, keeping inner text."""
    import re
    return re.sub(r'<cite[^>]*>(.*?)</cite>', r'\1', text, flags=re.DOTALL)


def main():
    print(f"Fetching latest IEEPA tariff refund news at {datetime.now(timezone.utc).isoformat()}...")

    response = call_claude_api()
    text = extract_text_from_response(response)

    if not text:
        print("Warning: No text content in API response")
        print("Full response:", json.dumps(response, indent=2)[:1000])
        sys.exit(1)

    items = parse_news_items(text)
    print(f"Found {len(items)} news items")

    # Validate items have required fields
    valid_items = []
    required_fields = ["headline", "summary"]
    for item in items:
        if all(item.get(f) for f in required_fields):
            # Strip citation markup from summaries
            item["summary"] = strip_citations(item["summary"])
            item["headline"] = strip_citations(item["headline"])
            # Ensure defaults
            item.setdefault("date", datetime.now(timezone.utc).strftime("%B %d, %Y").upper())
            item.setdefault("badge", "UPDATE")
            item.setdefault("category", "update")
            item.setdefault("source", "")
            valid_items.append(item)
        else:
            print(f"Skipping invalid item: {item}")

    news_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "items": valid_items
    }

    with open(NEWS_FILE, "w") as f:
        json.dump(news_data, f, indent=2)

    print(f"Wrote {len(valid_items)} items to {NEWS_FILE}")


if __name__ == "__main__":
    main()

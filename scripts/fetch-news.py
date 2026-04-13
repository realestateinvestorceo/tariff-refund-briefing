#!/usr/bin/env python3
"""
Fetch latest IEEPA tariff refund news using Claude API with web search,
compute a Go/No-Go business viability score, and write both to JSON files.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_FILE = os.path.join(BASE_DIR, "news-data.json")
SCORE_FILE = os.path.join(BASE_DIR, "score-data.json")

# --- Scoring constants ---

FACTOR_WEIGHTS = {
    "cape_complexity": 0.20,
    "window_remaining": 0.18,
    "market_readiness": 0.18,
    "legal_certainty": 0.15,
    "competition_density": 0.12,
    "fee_pressure": 0.10,
    "regulatory_risk": 0.07,
}

FACTOR_NAMES = {
    "cape_complexity": "CAPE System Friction",
    "window_remaining": "Time Window",
    "market_readiness": "Importer Unpreparedness",
    "legal_certainty": "Legal Stability",
    "competition_density": "Competitive Space",
    "fee_pressure": "Fee Sustainability",
    "regulatory_risk": "Regulatory Risk",
}

VERDICT_RANGES = [
    (30, "Don't start", "#DC2626"),
    (50, "Wait and watch", "#D97706"),
    (70, "Move cautiously", "#2563EB"),
    (85, "Go \u2014 move fast", "#059669"),
    (100, "Urgent \u2014 window closing", "#7C3AED"),
]

# --- News prompts ---

NEWS_SYSTEM_PROMPT = """You are a trade policy research assistant. Search the web for the latest news about IEEPA tariff refunds in the United States.

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

Return ONLY valid JSON — an array of objects. No markdown, no code fences. Return between 3 and 8 news items, ordered by importance/recency."""

# --- Scoring prompt ---

SCORE_SYSTEM_PROMPT = """You are a business opportunity analyst evaluating whether a SPECIFIC business — a mid-market intermediary that helps importers recover IEEPA tariff refunds from U.S. Customs and Border Protection (CBP) — is viable right now.

You will be given recent news context. You also have web search to verify and supplement.

Score each of the following 7 factors on a scale of 1-10, where 10 means the factor is MOST FAVORABLE for launching this intermediary business, and 1 means LEAST FAVORABLE.

FACTORS TO SCORE:

1. CAPE System Friction (cape_complexity)
   Score 10 = CAPE delayed, broken, or extremely complex — importers need help
   Score 1 = CAPE works perfectly, importers easily file alone

2. Time Window (window_remaining)
   Score 10 = Massive backlog, 6-12+ months of processing ahead
   Score 1 = Backlog nearly cleared, window essentially closed

3. Importer Unpreparedness (market_readiness)
   Score 10 = 80%+ of importers still unenrolled, widespread confusion
   Score 1 = Nearly all importers enrolled and handling refunds themselves

4. Legal Stability (legal_certainty)
   Score 10 = Courts strongly support refunds, no serious legal threats
   Score 1 = Major pending appeal could eliminate refund eligibility

5. Competitive Space (competition_density)
   Score 10 = Few competitors in mid-market, mostly expensive law firms
   Score 1 = Dozens of low-cost competitors, saturated market

6. Fee Sustainability (fee_pressure)
   Score 10 = 3-8% fees accepted, stable pricing
   Score 1 = Fees below 2%, free tools available, importers expect to pay nothing

7. Regulatory/Legislative Risk (regulatory_risk)
   Score 10 = Stable regulatory environment, no legislative threats
   Score 1 = Active legislation could eliminate refunds

RULES:
- Base every score on CURRENT, VERIFIABLE facts — not speculation
- Cite specific data points in reasoning (dates, percentages, case names)
- Each reasoning should be 2-3 sentences with concrete evidence
- If you cannot find current data for a factor, say so and score conservatively (4-6)

Return ONLY valid JSON (no markdown, no code fences):
{
  "factors": [
    {"id": "cape_complexity", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "window_remaining", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "market_readiness", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "legal_certainty", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "competition_density", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "fee_pressure", "score": <1-10>, "reasoning": "<2-3 sentences>"},
    {"id": "regulatory_risk", "score": <1-10>, "reasoning": "<2-3 sentences>"}
  ],
  "score_summary": "<One paragraph executive summary of overall opportunity viability>"
}"""


def strip_citations(text):
    """Remove <cite ...>...</cite> tags, keeping inner text."""
    return re.sub(r'<cite[^>]*>(.*?)</cite>', r'\1', text, flags=re.DOTALL)


def call_api(system_prompt, user_prompt, max_search_uses=10, timeout=120):
    """Call the Anthropic API with web search tool enabled."""
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_search_uses}],
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
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
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"API error {e.code}: {body}")
        return None


def extract_text(response):
    """Extract all text blocks from the Claude response."""
    if not response:
        return ""
    texts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            texts.append(block["text"])
    return "\n".join(texts)


def parse_json_array(text):
    """Parse a JSON array from text, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def parse_json_object(text):
    """Parse a JSON object from text, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def compute_score(factors):
    """Compute composite score, verdict, and enriched factor list."""
    enriched = []
    composite = 0.0
    for f in factors:
        fid = f.get("id", "")
        score = max(1, min(10, int(f.get("score", 5))))
        weight = FACTOR_WEIGHTS.get(fid, 0)
        contribution = ((score - 1) / 9.0) * weight * 100
        composite += contribution
        enriched.append({
            "id": fid,
            "name": FACTOR_NAMES.get(fid, fid),
            "score": score,
            "weight": weight,
            "weighted_contribution": round(contribution, 1),
            "reasoning": strip_citations(f.get("reasoning", "")),
        })

    composite = round(composite)
    verdict, color = "Wait and watch", "#D97706"
    for threshold, v, c in VERDICT_RANGES:
        if composite <= threshold:
            verdict, color = v, c
            break

    return composite, verdict, color, enriched


def detect_trend(new_composite):
    """Compare to previous score file and return trend direction."""
    try:
        with open(SCORE_FILE, "r") as f:
            old = json.load(f)
        old_score = old.get("composite_score", new_composite)
        delta = new_composite - old_score
        if delta > 3:
            return "improving"
        elif delta < -3:
            return "declining"
        return "stable"
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "stable"


# --- Main ---

def main():
    today = datetime.now(timezone.utc)
    date_str = today.strftime("%B %d, %Y")
    print(f"Fetching latest IEEPA tariff refund news at {today.isoformat()}...")

    # 1. Fetch news
    news_prompt = f"Search for the latest news and developments about IEEPA tariff refunds, the CBP CAPE system, and related Court of International Trade rulings. Focus on developments from the past week. Today's date is {date_str}."
    response = call_api(NEWS_SYSTEM_PROMPT, news_prompt)
    text = extract_text(response)

    if not text:
        print("Warning: No text content in news API response")
        sys.exit(1)

    items = parse_json_array(text)
    print(f"Found {len(items)} news items")

    valid_items = []
    for item in items:
        if item.get("headline") and item.get("summary"):
            item["summary"] = strip_citations(item["summary"])
            item["headline"] = strip_citations(item["headline"])
            item.setdefault("date", date_str.upper())
            item.setdefault("badge", "UPDATE")
            item.setdefault("category", "update")
            item.setdefault("source", "")
            valid_items.append(item)

    news_data = {"last_updated": today.isoformat(), "items": valid_items}
    with open(NEWS_FILE, "w") as f:
        json.dump(news_data, f, indent=2)
    print(f"Wrote {len(valid_items)} items to {NEWS_FILE}")

    # 2. Compute Go/No-Go score
    print("Computing Go/No-Go score...")
    news_context = "\n".join(
        f"- [{it.get('date','')}] {it.get('headline','')}: {it.get('summary','')}"
        for it in valid_items
    )
    score_prompt = (
        f"Today is {date_str}.\n\n"
        f"Here is today's news context about IEEPA tariff refunds:\n\n{news_context}\n\n"
        f"Using this context AND your own web search to verify and supplement, "
        f"score each of the 7 business viability factors for launching a mid-market "
        f"IEEPA tariff refund intermediary business."
    )

    score_response = call_api(SCORE_SYSTEM_PROMPT, score_prompt, max_search_uses=5, timeout=180)
    score_text = extract_text(score_response)

    if not score_text:
        print("Warning: No text content in scoring API response")
        return

    score_data = parse_json_object(score_text)
    if not score_data or not score_data.get("factors"):
        print("Warning: Could not parse scoring JSON")
        print("Response text:", score_text[:500])
        return

    composite, verdict, color, enriched = compute_score(score_data["factors"])
    trend = detect_trend(composite)

    final_score = {
        "last_updated": today.isoformat(),
        "composite_score": composite,
        "verdict": verdict,
        "verdict_color": color,
        "trend": trend,
        "score_summary": strip_citations(score_data.get("score_summary", "")),
        "factors": enriched,
    }

    with open(SCORE_FILE, "w") as f:
        json.dump(final_score, f, indent=2)
    print(f"Score: {composite}/100 — {verdict} (trend: {trend})")


if __name__ == "__main__":
    main()

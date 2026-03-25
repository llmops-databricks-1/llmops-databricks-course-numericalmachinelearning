import re

import requests

from profilr.config import get_secret


def _normalize_linkedin_url(url: str) -> str:
    """Normalize regional LinkedIn URLs to www.linkedin.com."""
    return re.sub(r"https?://[a-z]{2}\.linkedin\.com", "https://www.linkedin.com", url)


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False) -> dict:
    """
    Scrape a LinkedIn profile via scrapin.io API.
    Use mock=True to hit a static Gist (avoids burning API credits during testing).
    """
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/numericalmachinelearning/4654c82879446e27b07da85c51ab0e74/raw/4b79fc046d0bf53c55c65a82971e856a44118465/alessandro-scrapin"
        response = requests.get(linkedin_profile_url, timeout=10)
    else:
        linkedin_profile_url = _normalize_linkedin_url(linkedin_profile_url)
        response = requests.post(
            "https://api.scrapin.io/v1/enrichment/profile",
            params={"apikey": get_secret("SCRAPIN_API_KEY")},
            json={
                "linkedInUrl": linkedin_profile_url,
                "includes": {
                    "includeCompany": True,
                    "includeSummary": True,
                    "includeFollowersCount": True,
                    "includeSkills": True,
                    "includeExperience": True,
                    "includeEducation": True,
                },
            },
            timeout=10,
        )

    data = response.json().get("person")

    if data is None:
        return {}

    return {
        k: v
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["certifications"]
    }

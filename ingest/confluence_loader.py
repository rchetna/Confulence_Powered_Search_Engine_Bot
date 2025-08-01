import requests
import os
from dotenv import load_dotenv
load_dotenv()

def fetch_confluence_pages():
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    email = os.getenv("CONFLUENCE_EMAIL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    auth = (email, api_token)

    url = f"{base_url}/rest/api/content?type=page&limit=50&expand=body.storage"
    res = requests.get(url, auth=auth)

    pages = []
    if res.status_code == 200:
        data = res.json()
        for page in data.get("results", []):
            title = page["title"]
            body = page["body"]["storage"]["value"]
            pages.append({"title": title, "content": body})
    else:
        raise Exception("Failed to fetch Confluence data", res.status_code, res.text)
    return pages

import requests
from bs4 import BeautifulSoup

def fetch_text_from_url(url):
    """Fetch and extract text content from a webpage using BeautifulSoup."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")  # Extract text from <p> tags
            extracted_text = "\n".join([para.get_text() for para in paragraphs])
            return extracted_text.strip() if extracted_text else "No readable content found."
        else:
            return f"Failed to load: {url} (Status Code: {response.status_code})"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

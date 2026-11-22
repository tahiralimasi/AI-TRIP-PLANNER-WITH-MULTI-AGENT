
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing")

client = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query):
    try:
        response = client.search(
            query=query,
            max_results=5
        )

        results = []

        for i, r in enumerate(response.get("results", []), 1):
            title = r.get("title", "Unknown")
            url = r.get("url", "")
            snippet = r.get("content", "").strip()

            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

            results.append(
                f"{i}. {title}\n{url}\n{snippet}"
            )

        return "\n\n".join(results) if results else "No hotel results found."

    except Exception as e:
        print(f"Tavily search error: {e}")
        return f"Hotel search failed: {str(e)}"


import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

API_URL = "http://api.aviationstack.com/v1/flights"


def search_flights(query):

    if not API_KEY:
        return "AVIATIONSTACK_API_KEY is missing."

    try:
        params = {
            "access_key": API_KEY,
            "limit": 5
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:
            return f"AviationStack API error: {data['error']}"

        flights = []

        for flight in data.get("data", [])[:5]:

            airline = flight.get("airline") or {}
            departure = flight.get("departure") or {}
            arrival = flight.get("arrival") or {}

            flights.append(
                f"""
Airline: {airline.get('name', 'Unknown')}
Flight: {(flight.get('flight') or {}).get('iata', 'Unknown')}
Departure: {departure.get('airport', 'Unknown')}
Arrival: {arrival.get('airport', 'Unknown')}
Status: {flight.get('flight_status', 'Unknown')}
"""
            )

        return "\n".join(flights) if flights else "No flight results found."

    except requests.RequestException as e:
        return f"Flight API request failed: {str(e)}"

    except Exception as e:
        return f"Flight search error: {str(e)}"

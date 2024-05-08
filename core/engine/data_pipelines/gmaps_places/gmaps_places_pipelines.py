"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os
import requests

import googlemaps as gmaps

from logging_stack import logger
import asyncio


class GMapsPlacesRetrievalPipeline:
    def __init__(self):
        self.gmaps = gmaps.Client(os.getenv("GMAPS_PLACE_KEY"))

    def _get_place_details(self, place_id):
        # Request place details
        place_details = self.gmaps.place(place_id)

        # Extract reviews from the response
        reviews = []
        if "reviews" in place_details["result"]:
            for review in place_details["result"]["reviews"]:
                review_info = {
                    "author_name": review.get("author_name", ""),
                    "rating": review.get("rating", ""),
                    "text": review.get("text", ""),
                    "time": review.get("time", ""),
                }
                reviews.append(review_info)

        return reviews

    def search_nearby_places(
        self, latitute: float, longitude: float, radius: int, sector: str
    ):
        # Define search parameters
        location = (latitute, longitude)

        # Perform a nearby search
        places = self.gmaps.places_nearby(
            location, radius=radius, type=sector
        )  # radius in meters

        # Extract relevant information from the response
        nearby_places = []
        # sort_idxs = list (range(len(places['results']) if places["results"] else 0 ) )
        # sort order  = star*number of comments
        for place in places["results"]:
            place_info = {
                "name": place.get("name", ""),
                "comments": place.get(
                    "user_ratings_total", ""
                ),  # Total number of user ratings
                "stars": place.get("rating", ""),  # Average rating
                "vicinity": place.get("vicinity", ""),  # Address
                "location": place["geometry"]["location"],  # latitute and longitude
                "place_id": place.get("place_id", ""),  # Place ID
            }
            nearby_places.append(place_info)

        return nearby_places

from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient


def fetch_candle_data(mongo_uri, db_name, collection_name):
    """Fetch candle data from MongoDB."""
    # Get starting timestamp for May 20th, 2025 midnight UTC
    start_dt = datetime(2025, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
    start_timestamp = int(start_dt.timestamp() * 1000)  # assuming 't' is in ms
    query = {'t': {'$gte': start_timestamp}}

    with MongoClient(mongo_uri) as client:

        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find(query).sort('t', 1))

    return pd.DataFrame(data)

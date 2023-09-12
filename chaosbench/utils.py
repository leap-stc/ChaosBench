from datetime import datetime

def convert_time(timestamp, time_format='%Y-%m-%d'):
    "Convert native datetimens object to specific format"
    
    timestamp_s = timestamp / 1e9  # Convert nanoseconds to seconds
    dt = datetime.utcfromtimestamp(timestamp_s)
    
    day = dt.strftime(time_format)
    
    return day
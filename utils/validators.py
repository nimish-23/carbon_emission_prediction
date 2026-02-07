# Input validation

def validate_year_input(data: dict) -> tuple:
    # Check if data exists and has year
    if not data or "year" not in data:
        return (False, "Missing 'year' in request", None)
    
    year = data["year"]
    
    # Check if year is an integer
    if not isinstance(year, int):
        return (False, "'year' must be an integer", None)
    
    # Check if year is in valid range
    if year < 1965 or year > 2100:
        return (False, "Year out of supported range", None)
    
    return (True, None, year)

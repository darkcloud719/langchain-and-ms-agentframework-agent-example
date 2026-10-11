from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather for a given city."""

    weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C"
    }

    return weather_data.get(city, "Weather data not available for this city.")

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000
    )

    
from mcp.server.fastmcp import FastMCP
import json
import os
import sys

# Initialize FastMCP server
mcp = FastMCP("Clio Catalog Server")

# Load the catalog once into memory
CATALOG_FILE = os.path.join(os.path.dirname(__file__), "data", "catalog.json")

def load_catalog():
    if not os.path.exists(CATALOG_FILE):
        return {"artworks": []}
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

CATALOG = load_catalog()

@mcp.tool()
def get_artwork_details(beacon_id: str) -> str:
    """
    Get detailed information about an artwork based on its physical location beacon ID.
    
    Args:
        beacon_id: The unique identifier broadcasted by the bluetooth beacon near the artwork 
                   (e.g., 'vangogh_starry_night', 'davinci_mona_lisa', 'vermeer_pearl_earring').
    
    Returns:
        A JSON string containing the artwork's title, artist, year, medium, history, 
        visual description, and technique. Returns an error message if the beacon_id is not found.
    """
    for artwork in CATALOG.get("artworks", []):
        if artwork.get("beacon_id") == beacon_id:
            return json.dumps(artwork, indent=2)
    
    return json.dumps({"error": f"No artwork found for beacon_id '{beacon_id}'"})

if __name__ == "__main__":
    # Run the server using standard input/output streams
    print("Museum Catalog MCP Server is running...", file=sys.stderr)
    mcp.run(transport='stdio')

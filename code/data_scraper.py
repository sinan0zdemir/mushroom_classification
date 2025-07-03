from pyinaturalist.node_api import get_observation_species_counts, get_observations
from pathlib import Path
import requests
from tqdm import tqdm
import time

#Place = Italy
PLACE_ID = 6973
#Taxon = Mushrooms
#I
TAXON_ID = 50814
TOP_N = 20
PHOTOS_PER_SPECIES = 1000
OUTPUT_DIR = Path("inaturalist_images")
OUTPUT_DIR.mkdir(exist_ok=True)
PER_PAGE = 200
HEADERS = {
    "User-Agent": "inaturalist-image-scraper/1.0"
}

def sanitize_name(name):
    return name.replace("/", "_").replace(" ", "_")



def download_image(url, dest_path):
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

#Get top 20 species
print("Fetching top species...")
species_data = get_observation_species_counts(
    place_id=PLACE_ID,
    taxon_id=TAXON_ID,
    quality_grade="research",
    per_page=TOP_N
)["results"]


# Download images for each species
for sp in species_data:
    sp_id = sp["taxon"]["id"]
    sp_name = sanitize_name(sp["taxon"]["name"])
    print(f"\n==> Processing species: {sp_name} (ID {sp_id})")

    species_dir = OUTPUT_DIR / sp_name
    species_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1

    pbar = tqdm(total=PHOTOS_PER_SPECIES, desc=f"Downloading {sp_name}", unit="img")

    while downloaded < PHOTOS_PER_SPECIES:
        response = get_observations(
            taxon_id=sp_id,
            quality_grade="research",
            per_page=PER_PAGE,
            page=page,
            order_by="created_at",
            order="desc"
        )

        results = response.get("results", [])
        if not results:
            print("No more results.")
            break

        for obs in results:
            for photo in obs.get("photos", []):
                if downloaded >= PHOTOS_PER_SPECIES:
                    break
                url = photo.get("url", "").replace("square", "original")
                filename = f"{downloaded:04}.jpg"
                dest_path = species_dir / filename

                if download_image(url, dest_path):
                    downloaded += 1
                    pbar.update(1)

        page += 1

    pbar.close()
    print(f"✔ Downloaded {downloaded} images for {sp_name}")

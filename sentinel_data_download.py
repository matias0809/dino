# Use EODAG to download images
# https://github.com/CS-SI/eodag

from eodag import EODataAccessGateway
import os

# Register at Copernicus Dataspace to get credentials:
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#registering-oauth-client

# Then configure your credentials
# use cop_dataspace in stead of 'peps'
# https://eodag.readthedocs.io/en/latest/getting_started_guide/configure.html#configure-eodag


output_path = "/home/mnachour/master/dino/data"
os.makedirs(output_path, exist_ok=True)  # create it if it doesn't exist

# Initialize EODAG
dag = EODataAccessGateway()
dag.set_preferred_provider("cop_dataspace")

# You will get a huge list from us
product_ids = [
    'S2A_MSIL1C_20230306T104921_N0510_R051_T32VLK_20240820T000248'
]

for product_id in product_ids:
    # Define search parameters
    # Maybe possible to search multiple product ids at once, but I'm not sure
    search_results = dag.search(
        productType="S2_MSI_L1C",  # we use level 1C (not 2A)
        id=product_id
    )

    # Download products
    #search_results[0].download(output="./data")
    product_path = search_results[0].download(output=output_path)
    print(f"Downloaded product saved at: {product_path}")

# Some images might be "offline", meaning they are moved to cold storage
# You might need to order them back to "online" before you can download them.
# See how here:
# https://eodag.readthedocs.io/en/stable/notebooks/api_user_guide/8_download.html#Order-OFFLINE-products
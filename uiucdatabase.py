import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
ZIP_URL = "https://m-selig.ae.illinois.edu/ads/coord_seligFmt.zip"
DOWNLOAD_FOLDER = Path.home() / "Downloads" / "UIUC_Airfoils"
MAX_WORKERS = 10  # For parallel downloading if scraping is needed

def setup_directory(path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory exists: {path}")

def download_and_extract_zip(url, target_folder):

    print(f"\nAttempting to download ZIP archive from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print("Download successful. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_folder)
        
        print(f"Success! All files extracted to: {target_folder}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ZIP download failed (HTTP Error): {e}")
    except zipfile.BadZipFile:
        print("ZIP download failed (Corrupted file).")
    except Exception as e:
        print(f"ZIP download failed: {e}")
    
    return False

def download_single_file(file_url, target_folder):
    filename = file_url.split('/')[-1]
    save_path = target_folder / filename
    
    try:
        if save_path.exists():
            return f"Skipped (Exists): {filename}"

        response = requests.get(file_url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return f"Downloaded: {filename}"
    except Exception as e:
        return f"Failed: {filename} - {e}"

def scrape_and_download_individual_files(base_url, target_folder):
    print(f"\nHaaye Allah: {base_url}")
    
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    
        links = soup.find_all('a', href=True)
        dat_urls = []
        for link in links:
            href = link['href']
            if href.lower().endswith('.dat'):
                full_url = urljoin(base_url, href)
                dat_urls.append(full_url)

        dat_urls = list(set(dat_urls))
        total_files = len(dat_urls)
        print(f"Found {total_files} airfoil files. Starting parallel download...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(download_single_file, url, target_folder): url for url in dat_urls}
            
            count = 0
            for future in as_completed(future_to_url):
                count += 1
                result = future.result()
                if count % 50 == 0 or count == total_files:
                    print(f"Progress: {count}/{total_files} - Last Action: {result}")
                    
        print("\nKaam hogaya.")
        
    except Exception as e:
        print(f"Holy shit: {e}")

def main():
    print(" UIUC Airfoil Database Downloader ")
    setup_directory(DOWNLOAD_FOLDER)
    
    success = download_and_extract_zip(ZIP_URL, DOWNLOAD_FOLDER)
    
    if not success:
        scrape_and_download_individual_files(BASE_URL, DOWNLOAD_FOLDER)
    
    print(f"\nGo to downloads: {DOWNLOAD_FOLDER}")

if __name__ == "__main__":
    main()

'''
File:               batchDownloadImagery.py
Author:             Sean P. McGill
Email:              Sean.P.McGill@usace.army.mil
Date Last Updated:  05/15/2025
Description:        Batch download CorpsCam imagery
'''

import os
from datetime import datetime, timedelta
import requests
import concurrent.futures

# --- Configuration ---
studySite = 'FrfTower'
camera = 'cx'
startDate_str = '2026_6_14'
endDate_str = '2026_6_15'
imageProduct = 'snap' #snap
collectFrequency = 5  # In minutes
MAX_WORKERS = 6  # Number of parallel download threads.

#Choose 'obliques' or 'orthophotos'
imageType = 'orthophotos'  # Change this to 'orthophotos' to pull processed imagery

#Set the output location based on the image type and camera
baseOutLocation = './data/Argus/'
if imageType.lower() == 'orthophotos':
    outLocation = os.path.join(baseOutLocation, 'Orthophotos', camera.upper())
else:
    outLocation = os.path.join(baseOutLocation, 'Obliques', camera.upper())


def generate_download_tasks(start_dt, end_dt, freq_minutes):
    """
    Generates a list of download tasks (URL and save path) within the specified
    date range and time window (10:00-23:00).
    """
    print(f"Generating list of {imageType} to download...")
    tasks = []
    current_dt = start_dt
    while current_dt <= end_dt:
        if 10 <= current_dt.hour <= 23:
            dateString = current_dt.strftime('%Y_%m_%d')
            dateTimeString = current_dt.strftime('%Y%m%dT%H%M')

            # Branch URL generation based on image type
            if imageType.lower() == 'orthophotos':
                url = (
                    f"https://coastalimaging.erdc.dren.mil/{studySite}/Processed/Orthophotos/{camera}/{dateString}/"
                    f"{dateTimeString}00Z.{studySite}.{camera}.{imageProduct}.jpg"
                )
            else:
                url = (
                    f"https://coastalimaging.erdc.dren.mil/{studySite}/Raw/Obliques/{camera}/{dateString}/"
                    f"{dateTimeString}00Z.{studySite}.{camera}.{imageProduct}.jpg"
                )

            saveName = f"{dateTimeString}00Z.{studySite}.{camera}.{imageProduct}.jpg"
            fpath = os.path.join(outLocation, saveName)

            if not os.path.isfile(fpath):
                tasks.append({'url': url, 'path': fpath})

        current_dt += timedelta(minutes=freq_minutes)

    print(f"Generated {len(tasks)} tasks for new images.")
    return tasks


def download_image(task, session):
    """
    Worker function to download a single image using a shared requests Session.
    """
    url = task['url']
    path = task['path']
    try:
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        if len(response.content) > 1000:
            with open(path, 'wb') as f:
                f.write(response.content)
            return url, "Success"
        else:
            return url, "Skipped (File size too small)"

    except requests.exceptions.RequestException as e:
        return url, f"Failed ({type(e).__name__})"


def main():
    """Main execution function."""
    if not os.path.exists(outLocation):
        os.makedirs(outLocation)
        print(f"Created output directory: {outLocation}")

    start_dt = datetime.strptime(startDate_str.replace('_', '') + '1000', '%Y%m%d%H%M')
    end_dt = datetime.strptime(endDate_str.replace('_', '') + '2300', '%Y%m%d%H%M')

    tasks = generate_download_tasks(start_dt, end_dt, collectFrequency)

    if not tasks:
        print("No new images to download. All files may already exist.")
        return

    print(f"Starting download of {len(tasks)} images with {MAX_WORKERS} parallel workers...")

    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(download_image, task, session): task for task in tasks}

            success_count = 0
            fail_count = 0

            for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
                url, status = future.result()
                if "Success" in status:
                    success_count += 1
                else:
                    fail_count += 1

                progress = (i + 1) / len(tasks) * 100
                print(
                    f"Progress: {progress:.2f}% ({i + 1}/{len(tasks)}) | Success: {success_count} | Failed: {fail_count}",
                    end='\r')

    print(f"\n\nDownload complete.                                  ")
    print(f"Successfully downloaded: {success_count} files.")
    print(f"Failed or skipped: {fail_count} files.")
    print(
        "Note: A high number of 'Failed' statuses is expected, as images are likely not generated for every 15-minute interval.")


if __name__ == "__main__":
    main()
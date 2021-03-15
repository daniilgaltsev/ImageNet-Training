"""A module containing function to download images."""


import asyncio
import os
from pathlib import Path
from typing import Dict, List, Union

import aiofiles
import aiohttp


async def download_image(
    url: str,
    filename: Union[Path, str],
    semaphore: asyncio.Semaphore,
    timeout: float
) -> bool:
    """Download an image from a given url to a given path.

    Args:
        url: A url from which to download.
        filename: A path to the file where to save the image..
        semaphore: A semaphore to limit concurrent download-writes.
        timeout: Time until download is abandoned.

    Returns:
        A boolean, which indicates if download-write was successful.
    """
    if os.path.exists(filename):
        return True
    try:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout, raise_for_status=True) as response:
                    if response.status != 200:
                        return False
                    f = await aiofiles.open(filename, "wb+")
                    await f.write(await response.read())
                    await f.close()
                    return True
    except Exception as e:  # pylint: disable=broad-except
        print(f"Unable to download image {url} to {filename} due to {e}.")
        return False


async def download_synset_images(
    images_data_dirname: Path,
    urls: List[str],
    synset: str,
    n_to_download: int,
    semaphore: asyncio.Semaphore,
    timeout: float
) -> int:
    """Downloads images using given urls for a synset.

    Args:
        images_data_dirname: A path where to save images.
        urls: A list of urls for downloading.
        synset: A name of the synset for which images are downloaded.
        n_to_download: A number of images to download.
        semaphore: A semaphore to limit concurrent download-writes.
        timeout: Time until download is abandoned.

    Returns:
        A number of downloaded images.
    """
    urls.sort(key=lambda url: url.find('flickr') == -1)
    downloaded = 0
    last_tried = 0
    while downloaded < n_to_download and last_tried < len(urls):
        end = last_tried + (n_to_download - downloaded)
        print(f"{synset}: {downloaded}/{n_to_download} next batch {last_tried}-{end}")
        results = await asyncio.gather(
            *[download_image(
                urls[idx],
                images_data_dirname / "{}_{}.jpg".format(synset, idx),
                semaphore,
                timeout
            ) for idx in range(last_tried, end)]
        )
        for res in results:
            downloaded += res
        last_tried = end
    print(f"{synset}: done, downloaded {downloaded}/{n_to_download}.")
    return downloaded


async def download_subsampled_images(
    images_dirname: Path,
    synsets: List[str],
    synsets_to_urls: Dict[str, List[str]],
    images_per_class: int,
    max_concurrent: int,
    timeout: float
) -> List[int]:
    """Downloads images using given urls.

    Args:
        images_dirname: A path where to save images.
        synsets: A list of synsets for which to download images.
        synsets_to_urls: A dict of synsets to their corresponding lists of urls.
        images_per_class: A number of images per synset/class to download.
        max_concurrent: A maximum number of concurrent image download-writes.
        timeout: Time until download is abandoned.

    Returns:
        A list describing number of images per synsets that were downloaded.
    """
    images_dirname.mkdir(exist_ok=True, parents=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    result = await asyncio.gather(
        *[download_synset_images(
            images_dirname,
            synsets_to_urls[synset],
            synset,
            images_per_class,
            semaphore,
            timeout) for synset in synsets]
    )
    return list(result)

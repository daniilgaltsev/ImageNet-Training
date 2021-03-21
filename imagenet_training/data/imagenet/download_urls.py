"""Module containing function to download image urls."""


import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp


GET_SYNSET_URL_API = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}"


async def _download_urls_for_synset(
    synset: str,
    semaphore: asyncio.Semaphore
) -> Tuple[str, Optional[List[str]]]:
    """Downloads a list of urls for a given synset.

    Args:
        synset: a synset for which to download urls.
        semaphore: a semaphore for blocking a request.
    """
    image_urls = []
    url = GET_SYNSET_URL_API.format(synset)
    try:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url=url) as response:
                    resp = await response.read()
    except Exception as e:  # pylint: disable=broad-except
        print(f"Unable to get url {url} due to {e}.")
        return synset, None

    for line in resp.decode().split('\n'):
        url = line.rstrip()
        if len(url) == 0:
            break
        image_urls.append(url)
    return synset, image_urls


async def _download_image_urls(
    urls_filename: Union[Path, str],
    synsets: List[str],
    max_concurrent: int = 50,
    rewrite: bool = False
) -> Dict[str, Optional[List[str]]]:
    """Downloads urls for each synset and saves them in json format in a given path.

    Args:
        urls_filename: a path to the file where to save the urls.
        synsets: a list of synsets for which to download urls.
        max_concurrent (optional): a maximum number of concurrent requests.
        rewrite (optional): if True, will download new urls even if file exists.
    """
    if (not rewrite) and os.path.exists(urls_filename):
        with open(urls_filename, "r") as f:
            return json.load(f)
    raise NotImplementedError("The ImageNet site was updated and there is no longer access to lists of urls by synset.")
    semaphore = asyncio.Semaphore(max_concurrent)  # pylint: disable=unreachable
    synsets_to_urls = dict(await asyncio.gather(*[_download_urls_for_synset(synset, semaphore) for synset in synsets]))
    with open(urls_filename, "w") as f:
        json.dump(synsets_to_urls, f)
    print(len(synsets_to_urls))
    return synsets_to_urls


def download_image_urls(
    urls_filename: Union[Path, str],
    synsets: List[str],
    max_concurrent: int = 50,
    rewrite: bool = False
) -> Dict[str, Optional[List[str]]]:
    """Downloads urls for each synset and saves them in json format in a given path.

    Args:
        urls_filename: a path to the file where to save the urls.
        synsets: a list of synsets for which to download urls.
        max_concurrent (optional): a maximum number of concurrent requests.
        rewrite (optional): if True, will download new urls even if file exists.
    """
    print("Downloading image urls.")
    synsets_to_urls = asyncio.run(_download_image_urls(urls_filename, synsets, max_concurrent, rewrite))
    return synsets_to_urls


def parse_synset_mapping(
    synset_mapping_filename: Union[Path, str]
) -> Tuple[List[str], Dict[str, str]]:
    """Parses the synset mapping file.

    Args:
        synset_mapping_filename: a path to the synset mapping file.

    Returns:
        A tuple containing a list of synsets and a dict from synset to its name.
    """
    synset_mapping = {}
    synsets = []
    with open(synset_mapping_filename, "r") as f:
        for line in f:
            synset, name = line.rstrip().split(' ', maxsplit=1)
            synsets.append(synset)
            synset_mapping[synset] = name
    return synsets, synset_mapping


def get_synset_stats(
    synsets_to_urls: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Calculate number of urls and number of flickr urls for each synset.

    Args:
        synsets_to_urls: a dict mapping a synset to a list of urls.
    """
    stats = {}
    for synset in synsets_to_urls:
        urls = synsets_to_urls[synset]
        n_urls = len(urls)
        n_flickr_urls = 0
        for url in urls:
            if url.find('flickr') != -1:
                n_flickr_urls += 1
        stats[synset] = {"n_urls": n_urls, "n_flickr_urls": n_flickr_urls}
    return stats

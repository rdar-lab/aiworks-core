import base64
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .prompt_compression import compress_prompt
from ..models import (
    LLMConfiguration,
)
from ..utils import async_to_sync

logger = logging.getLogger(__name__)

TEXT_EXT = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".txt",
    ".text",
    ".md",
    ".yaml",
    ".yml",
    ".xml",
    ".svg",
}

BASE64_REGEX = re.compile(
    r'^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$'
)

# Precompiled regex for sanitizing filenames (keeps alphanumeric, dots, underscores, hyphens)
_FILENAME_SANITIZE_RE = re.compile(r"[\x00/\\:*?\"<>|]")

# All private / special-use IP networks that must never be reachable from an MCP server URL.
# Using explicit ranges rather than ipaddress property helpers (is_private, is_reserved, …) makes
# coverage version-independent and easier to audit.
_BLOCKED_IP_NETWORKS = [
    # IPv4 — private / special-use (RFC1918, RFC5735, RFC5737, RFC6598, RFC6890, …)
    ipaddress.ip_network('0.0.0.0/8'),  # "This" network (RFC1122)
    ipaddress.ip_network('10.0.0.0/8'),  # RFC1918 Class A private
    ipaddress.ip_network('100.64.0.0/10'),  # RFC6598 shared address (carrier-grade NAT)
    ipaddress.ip_network('127.0.0.0/8'),  # Loopback (RFC1122)
    ipaddress.ip_network('169.254.0.0/16'),  # Link-local / APIPA / cloud IMDS (RFC3927)
    ipaddress.ip_network('172.16.0.0/12'),  # RFC1918 Class B private
    ipaddress.ip_network('192.0.0.0/24'),  # RFC6890 IETF protocol assignments
    ipaddress.ip_network('192.0.2.0/24'),  # RFC5737 TEST-NET-1 (documentation)
    ipaddress.ip_network('192.168.0.0/16'),  # RFC1918 Class C private
    ipaddress.ip_network('198.18.0.0/15'),  # RFC2544 benchmarking
    ipaddress.ip_network('198.51.100.0/24'),  # RFC5737 TEST-NET-2 (documentation)
    ipaddress.ip_network('203.0.113.0/24'),  # RFC5737 TEST-NET-3 (documentation)
    ipaddress.ip_network('240.0.0.0/4'),  # Reserved / Class E (RFC1112)
    ipaddress.ip_network('255.255.255.255/32'),  # Broadcast
    # IPv6 — private / special-use
    ipaddress.ip_network('::1/128'),  # IPv6 loopback (RFC4291)
    ipaddress.ip_network('::ffff:0:0/96'),  # IPv4-mapped IPv6 (RFC4291)
    ipaddress.ip_network('64:ff9b::/96'),  # IPv4/IPv6 translation (RFC6052)
    ipaddress.ip_network('fc00::/7'),  # Unique Local (ULA, RFC4193) — includes fd00::/8 (AWS IPv6 IMDS)
    ipaddress.ip_network('fe80::/10'),  # IPv6 link-local (RFC4291)
]


def compress_if_needed(original_raw, cached_compressed, save_compressed_callback):
    # If no compression is activated, return original
    cfg = LLMConfiguration.get_solo()
    if not cfg.compress_prompts:
        return original_raw

    if cached_compressed:
        return cached_compressed

    compressed = async_to_sync(compress_prompt)(original_raw)
    if compressed and compressed != original_raw:
        save_compressed_callback(compressed)
    return (
            compressed or original_raw
    )  # Fallback to original if compression fails or returns empty


async def extract_title_from_response(result) -> Any:
    title = result.strip()
    # Remove surrounding quotes if the LLM added them
    if len(title) >= 2 and title[0] in ('"', "'") and title[-1] == title[0]:
        title = title[1:-1].strip()
    logger.info("_extract_title_from_response | title=%r", title[:100])
    # Safety-cap: prompt requests ≤10 words; 100 chars covers that with room to spare
    return title[:100]


def get_file_ext(name):
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def is_text_file(name):
    return get_file_ext(name) in TEXT_EXT


def is_base64(s: str) -> bool:
    if not s:
        return False
    cleaned_s = "".join(s.split())
    if not BASE64_REGEX.fullmatch(cleaned_s):
        return False
    try:
        base64.decodebytes(s.encode("utf-8"))
        return True
    except Exception:
        return False


def validate_ssrf_safe_url(url: str) -> None:
    """
    Validate that a URL is safe to use as an outbound HTTP target.

    Raises ``ValueError`` if the URL:
    - Uses a scheme other than http or https
    - Resolves to a loopback, private, link-local, or otherwise reserved IP
    - Resolves to a known cloud-provider metadata endpoint

    Note: this performs a DNS lookup at registration/validation time to catch
    hostnames that resolve to internal IPs. It does not defend against DNS
    rebinding after registration; use a network-level egress policy for that.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc

    if parsed.scheme not in ('http', 'https'):
        raise ValueError(
            f"Unsupported scheme '{parsed.scheme}'; only http and https are allowed for MCP servers"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("MCP server URL must include a hostname")

    try:
        _, _, ip_list = socket.gethostbyname_ex(hostname)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve MCP server hostname '{hostname}': {exc}") from exc

    if not ip_list:
        raise ValueError(f"No IP addresses resolved for hostname '{hostname}'")

    for ip_str in ip_list:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            # Fail safe: if we cannot parse an address we must not silently allow it.
            raise ValueError(
                f"MCP server hostname '{hostname}' resolved to an unparseable address: {ip_str!r}"
            )

        matched = next(
            (net for net in _BLOCKED_IP_NETWORKS if ip in net),
            None,
        )
        if matched is not None:
            raise ValueError(
                f"MCP server URL resolves to a blocked IP address ({ip_str}) "
                f"in restricted network {matched}"
            )


def base_url(url: str) -> str:
    """Return the scheme+host (no path) of *url*."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def safe_get_with_ssrf_check(
        client: httpx.AsyncClient,
        url: str,
        max_redirects: int = 3,
) -> httpx.Response:
    """
    GET *url* following up to *max_redirects* redirects, with SSRF validation
    applied to every redirect target before following it.

    The ``client`` must be created with ``follow_redirects=False`` so that
    httpx does not silently follow redirects to unvalidated hosts.

    Raises ``httpx.TooManyRedirects`` if the redirect chain exceeds
    *max_redirects* hops.
    """
    resp = None
    for _ in range(max_redirects + 1):
        resp = await client.get(url)
        if not resp.is_redirect:
            return resp
        location = resp.headers.get("location", "")
        if not location:
            return resp
        location = urljoin(url, location)
        validate_ssrf_safe_url(location)
        url = location
    raise httpx.TooManyRedirects(
        f"Exceeded {max_redirects} redirects while probing {url!r}", request=(resp.request if resp else None)
    )


def make_safe_filename(filename: str) -> str:
    return _FILENAME_SANITIZE_RE.sub("_", filename)

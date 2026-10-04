import socket

import pytest

from tech_tree_arena.runtime.linux_sandbox import connect_destination


@pytest.mark.parametrize("wire_request", [
    b"GET https://chatgpt.com/ HTTP/1.1", b"CONNECT localhost:443 HTTP/1.1",
    b"CONNECT 127.0.0.1:443 HTTP/1.1", b"CONNECT chatgpt.com:80 HTTP/1.1",
    b"CONNECT chatgpt.com.attacker.test:443 HTTP/1.1", b"CONNECT user@chatgpt.com:443 HTTP/1.1",
    b"CONNECT [::1]:443 HTTP/1.1", b"CONNECT chatgpt.com:443 HTTP/2",
])
def test_proxy_rejects_unapproved_requests_without_dns(monkeypatch, wire_request):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS must not run"))
    with pytest.raises(ValueError):
        connect_destination(wire_request + b"\r\n\r\n")


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1"])
def test_proxy_rejects_private_dns_answers(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))])
    with pytest.raises(ValueError, match="Non-public"):
        connect_destination(b"CONNECT chatgpt.com:443 HTTP/1.1\r\n\r\n")


def test_proxy_returns_only_validated_public_addresses(monkeypatch):
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: addresses)
    assert connect_destination(b"CONNECT chatgpt.com:443 HTTP/1.1\r\n\r\n") == addresses

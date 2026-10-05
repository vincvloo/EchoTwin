"""The laptop's addresses and the self-signed certificate (phones need HTTPS for the camera and motion sensors)."""
import datetime
import ipaddress
import socket

from . import config


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def all_ips() -> list[str]:
    """Every IPv4 address of this laptop the phone might use: Wi-Fi, USB tethering, hotspot."""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    main_ip = lan_ip()
    if main_ip not in ips and not main_ip.startswith("127."):
        ips.insert(0, main_ip)
    return ips or ["127.0.0.1"]


def ensure_cert(ip: str):
    """Self-signed cert for localhost + this LAN IP."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    cert_p, key_p, ip_p = config.CERTS / "cert.pem", config.CERTS / "key.pem", config.CERTS / "ip.txt"
    ips = sorted(set(all_ips() + [ip]))
    if cert_p.exists() and ip_p.exists() and ip_p.read_text() == ",".join(ips):
        return str(cert_p), str(key_p)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "echotwin")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                                                       + [x509.IPAddress(ipaddress.ip_address(a)) for a in ips]), False)
            .sign(key, hashes.SHA256()))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                        serialization.NoEncryption()))
    ip_p.write_text(",".join(ips))
    return str(cert_p), str(key_p)

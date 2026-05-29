import os

from dotenv import load_dotenv
from requests import Session
from requests.auth import HTTPBasicAuth
from zeep import Client, Settings
from zeep.transports import Transport

load_dotenv()

USERNAME = os.getenv("RAILWAY_USERNAME")
PASSWORD = os.getenv("RAILWAY_PASSWORD")
if not USERNAME:
    USERNAME = os.getenv("USERNAME")
if not PASSWORD:
    PASSWORD = os.getenv("PASSWORD")


WSDL = "https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl"


def get_ojp_client():
    """Return a Zeep Client configured with HTTP Basic Authentication and
    settings to ignore sequence order."""
    session = Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)

    # Create settings that ignore sequence order and disable strict mode
    settings = Settings(
        strict=False, xml_huge_tree=True, xsd_ignore_sequence_order=True
    )

    transport = Transport(session=session)
    client = Client(WSDL, transport=transport, settings=settings)
    return client

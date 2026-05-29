import os
from zeep import Client, Settings
from requests import Session
from requests.auth import HTTPBasicAuth
from zeep.transports import Transport
from dotenv import load_dotenv

load_dotenv()

USERNAME = "wwang"
PASSWORD = "?i92S6"


WSDL = "https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl"

def get_ojp_client():
    """Return a Zeep Client configured with HTTP Basic Authentication and 
    settings to ignore sequence order."""
    session = Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)
    
    # Create settings that ignore sequence order and disable strict mode
    settings = Settings(strict=False, xml_huge_tree=True, xsd_ignore_sequence_order=True)
    
    transport = Transport(session=session)
    client = Client(WSDL, transport=transport, settings=settings)
    return client
import os
from zeep import Client
from requests import Session
from requests.auth import HTTPBasicAuth
from zeep.transports import Transport
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


WSDL = "https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl"

def get_ojp_client():
    session = Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)

    transport = Transport(session=session)
    client = Client(WSDL, transport=transport)
    return client
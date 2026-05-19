from zeep import Client
from requests import Session
from requests.auth import HTTPBasicAuth
from zeep.transports import Transport


USERNAME = "wwang"
PASSWORD = "?i92S6"


WSDL = "https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl"

def get_ojp_client():
    session = Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)

    transport = Transport(session=session)
    client = Client(WSDL, transport=transport)
    return client
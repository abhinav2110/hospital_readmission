import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
 
names=["ram","shyam"]
usernames=["ra","sha"]
passwords=["121","122"]
hash=Hasher(passwords).generate()
fp=Path(__file__).parent/"hash.pkl"
with fp.open("wb") as file:
    pickle.dump(hash,file)
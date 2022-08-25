import re
import io
import zlib
import base64
import hashlib
from re import Match
from pathlib import Path 
from cryptography.fernet import Fernet 

def _generate_model_secret(s):
    n = 4
    if "-" not in s:
        s = "-".join(s[i:i + n] for i in range(0, len(s), n))

    d = zlib.crc32(s.encode())
    pat = re.compile(r"-")

    def repl(m: Match) -> str:
        idx = m.start() // (n + 1)
        b = (idx % 2) == 0
        if b:
            c = "a"
            r = 26
        else:
            c = "0"
            r = 10
        x = chr(ord(c) + (d & ((1 << idx) % r)))
        return x

    return hashlib.sha256(pat.sub(repl, s).encode("UTF-8")).digest()


def get_mac_password(do_decrpt: bool = False):
    if not do_decrpt:
        from pystar360.utilities._logger import d_logger
        d_logger.debug(">>> No decrption.")
        return None 
    from pystar360.utilities.machine import get_machine_code
    mac_code: str = get_machine_code()
    assert isinstance(mac_code, str), ">>> Mac code must be str"
    password = base64.urlsafe_b64encode(_generate_model_secret(mac_code))
    print(">>> Use decrption.")
    return password 


def decrpt_content_from_filepath(filepath, key, encrp_exts=".pystar"):
    filepath = Path(filepath)
    
    content = ""
    # check file exits and ensure that it's a file 
    if filepath.exists() and filepath.is_file():
        # read binary data from file
        with open(str(filepath), "rb") as f:
            encrypted_data = f.read()
        
        # if its extention is .pystar, decrpt
        if filepath.suffix == encrp_exts:
            encrypted_data = base64.urlsafe_b64encode(encrypted_data) #TODO
            content = Fernet(key).decrypt(encrypted_data)
        
        # get content
        content = io.BytesIO(content)
        content.seek(0)
    else:
        raise FileNotFoundError(f">>> {filepath} not found.")
    return content
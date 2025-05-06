from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
import jwt
import logging
from jwt import PyJWTError
from src.utils.database import get_auth_db 

SECRET_KEY = os.getenv("TOKEN_SECRET_KEY")
ALGORITHM = os.getenv("TOKEN_ALGORITHM", "HS256")
ISSUER = os.getenv("TOKEN_ISSUER", "SKYWORXToken")
EXPECTED_AUDIENCE = os.getenv("TOKEN_AUDIENCE", "SKYWORXAudience")  # Add this line

auth_scheme = HTTPBearer()

def verify_jwt(token: str):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=EXPECTED_AUDIENCE  # This must match the "aud" in your JWT
        )
        return payload
    except PyJWTError as e:
        logging.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),
    db: Session = Depends(get_auth_db)
):
    """
    Extracts and verifies the token from the request header.
    """
    return verify_jwt(credentials.credentials)

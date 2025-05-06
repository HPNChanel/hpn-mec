from pydantic import BaseModel
from app.schemas.user import UserResponse

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenResponse(BaseModel):
    token: Token
    user: UserResponse

from typing import Any, Optional
from pydantic import BaseModel

class ResponseModel(BaseModel):
    status: str
    data: Any = None
    message: Optional[str] = None 
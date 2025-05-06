from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TimestampSchema(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(description="A short summary of the person")
    facts: list[str] = Field(description="Interesting facts about the person")

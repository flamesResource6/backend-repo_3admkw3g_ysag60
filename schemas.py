"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (kept for reference)
class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Study assistant schemas
class MemoryNote(BaseModel):
    """
    Notes and facts saved by the user for long-term recall.
    Collection name: "memorynote"
    """
    content: str = Field(..., description="The text to remember (any language)")
    language: Optional[str] = Field(None, description="ISO code or language name if known")
    tags: List[str] = Field(default_factory=list, description="Optional tags like 'math', 'history'")

class ConversationTurn(BaseModel):
    """
    A single conversational exchange for history.
    Collection name: "conversationturn"
    """
    role: str = Field(..., description="'user' or 'assistant'")
    text: str = Field(..., description="Message content (any language)")
    session_id: Optional[str] = Field(None, description="Client-provided session identifier")
    language: Optional[str] = Field(None, description="Language of the message if known")


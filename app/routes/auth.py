# main.py
import os
from typing import List, Optional
from datetime import date
from fastapi import FastAPI, HTTPException, status, APIRouter
from pydantic import BaseModel, Field
from pymongo import MongoClient, errors
from bson.objectid import ObjectId

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "chefdb"
COLLECTION_NAME = "users"

# ---------- DB Setup ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db[COLLECTION_NAME]

# ensure unique index on unique_id
try:
    users_col.create_index("unique_id", unique=True)
except errors.OperationFailure:
    # index may already exist or insufficient privileges — ignore for now
    pass

# ---------- Pydantic Models ----------
class UserCreate(BaseModel):
    unique_id: str = Field(..., example="user123")
    name: str = Field(..., example="John Doe")
    password: str = Field(..., example="plainpassword")   # stored as-is (insecure)
    dob: Optional[date] = Field(None, example="2000-01-01")
    gender: Optional[str] = Field(None, example="male")
    messages: Optional[List[str]] = Field(default_factory=list, example=["hi"])

class UserOut(BaseModel):
    id: str
    unique_id: str
    name: str
    dob: Optional[date]
    gender: Optional[str]
    messages: List[str]

class LoginModel(BaseModel):
    unique_id: str
    password: str

class MessageIn(BaseModel):
    message: str

# ---------- App ----------
app = APIRouter()

# helper serializer
def serialize_user(doc) -> UserOut:
    return UserOut(
        id=str(doc.get("_id")),
        unique_id=doc.get("unique_id"),
        name=doc.get("name"),
        dob=doc.get("dob"),
        gender=doc.get("gender"),
        messages=doc.get("messages", []),
    )

# ---------- Routes ----------
@app.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate):
    """
    Create a new user. unique_id must be unique.
    Password will be stored as plain text (insecure) as requested.
    """
    doc = user.model_dump()
    # ensure messages exists
    if "messages" not in doc or doc["messages"] is None:
        doc["messages"] = []

    try:
        result = users_col.insert_one(doc)
    except errors.DuplicateKeyError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="unique_id already exists")
    created = users_col.find_one({"_id": result.inserted_id})
    return serialize_user(created)

@app.post("/login", response_model=UserOut)
def login(payload: LoginModel):
    """
    Simple login by matching unique_id + password (plain text check).
    Returns user (without password).
    """
    doc = users_col.find_one({"unique_id": payload.unique_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")
    if doc.get("password") != payload.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")
    return serialize_user(doc)

@app.get("/users/{unique_id}", response_model=UserOut)
def get_user(unique_id: str):
    doc = users_col.find_one({"unique_id": unique_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return serialize_user(doc)

@app.post("/users/{unique_id}/messages", response_model=UserOut)
def add_message(unique_id: str, payload: MessageIn):
    """
    Append a message string to the user's messages array.
    """
    update_result = users_col.find_one_and_update(
        {"unique_id": unique_id},
        {"$push": {"messages": payload.message}},
        return_document=True
    )
    if not update_result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    # find_one_and_update with return_document=True returns BEFORE doc in PyMongo by default
    # To simplify, fetch fresh doc:
    doc = users_col.find_one({"unique_id": unique_id})
    return serialize_user(doc)

@app.delete("/users/{unique_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(unique_id: str):
    res = users_col.delete_one({"unique_id": unique_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return None

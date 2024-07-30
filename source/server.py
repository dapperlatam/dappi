from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import ai as ai

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ask/{question}")
def read_item(question: Union[str, None] = None):
    return ai.respondedor(question)

@app.get("/update")
async def update():
    ai.actualizar()
    return {"status": "success"}

@app.get("/update-all")
async def update():
    ai.actualizar(True)
    return {"status": "success"}
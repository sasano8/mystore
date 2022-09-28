import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

ROOT_DIR: str
AS_TEMP: bool


class Config(BaseModel):
    ROOT_DIR: str = ""
    AS_TEMP: bool = False

    def setup(self):
        global ROOT_DIR
        global AS_TEMP
        ROOT_DIR = self.ROOT_DIR
        AS_TEMP = self.AS_TEMP


@app.on_event("startup")
def startup():
    global ROOT_DIR
    if ROOT_DIR is None:
        raise RuntimeError()


@app.on_event("shutdown")
def shutdown():
    ...


def get_root_dir():
    import os

    os.environ.get("MODEL_DIR")

    if not os.path.exists(dir):
        os.mkdir(dir)

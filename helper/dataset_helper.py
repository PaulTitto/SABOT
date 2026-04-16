import asyncio
import os.path
import re
from typing import Optional

DATA_ROOT = "./data/"
def extract_doc_id(content: str) -> Optional[str]:
    match = re.search(r'^id:\s*(\S+)', content, re.MULTILINE)
    return  match.group(1) if match else None


async def insert_document(rag, root_dir: str):
    if not os.path.exists(root_dir):
        print(f"Folder {root_dir} doesn't exist")
        return
    files = []
    for dir_path, _, file_names in os.walk(root_dir):
        for f in file_names:
            if f.endswith(".txt"):
                files.append(os.path.join(dir_path, f))
    print(f"Found {len(files)} file .txt")
    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            content = fp.read().strip()
        if not content:
            continue
        doc_id = extract_doc_id(content)
        try:
            await rag.ainsert(content)
            print(f"Inserted {doc_id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Document {doc_id} already exists n Skipped ")
            else:
                print(f"Error {doc_id}: {e}")
        await  asyncio.sleep(0.2)
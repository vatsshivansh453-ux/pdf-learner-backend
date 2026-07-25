import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.pdf_reader import extract_text_from_pdf
from utils.text_splitter import split_text_into_chunks
from utils.embedding_test import create_embeddings

from utils.memory import (
    get_chat_history,
    add_message,
    clear_memory,
    create_session,
    get_sessions,
    get_session_owner,
    delete_session,
    create_table
)

from utils.vector_store import (
    create_faiss_index,
    save_vector_store,
    load_vector_store,
    add_embeddings
)

from utils.rag import (
    generate_answer,
    stream_answer
)

from utils.llm import (
    generate_emoji_summary,
    generate_cheat_sheet
)

from utils.auth import (
    oauth,
    create_access_token,
    upsert_user_from_google,
    upsert_user_from_github,
    get_current_user,
    get_optional_user,
    COOKIE_NAME,
    FRONTEND_URL,
    JWT_EXPIRE_DAYS,
)

load_dotenv()

# =====================================================
# APP
# =====================================================

app = FastAPI(title="PDF-LEARNER API")

# Required by Authlib to store OAuth state/nonce between redirect + callback
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "dev-session-secret-change-me"),
    same_site="lax",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_table()

faiss_index, pdf_chunks = load_vector_store()

print("=" * 60)
print("VECTOR STORE LOADED")
print("Chunks :", len(pdf_chunks))
print("Vectors:", faiss_index.ntotal if faiss_index else 0)
print("=" * 60)

# =====================================================
# MODELS
# =====================================================

class QuestionRequest(BaseModel):
    question: str
    session_id: str


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {

        "project": "PDF-LEARNER API",

        "developer": "Shivansh Vats",

        "status": "Running"

    }


# =====================================================
# AUTH: GOOGLE
# =====================================================

@app.get("/auth/login/google")
async def login_google(request: Request):

    if "google" not in oauth._clients:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET."
        )

    redirect_uri = request.url_for("auth_callback_google")

    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback/google")
async def auth_callback_google(request: Request):

    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo")

    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token)

    user = await upsert_user_from_google(userinfo)

    return _issue_session_and_redirect(user["id"])


# =====================================================
# AUTH: GITHUB
# =====================================================

@app.get("/auth/login/github")
async def login_github(request: Request):

    if "github" not in oauth._clients:
        raise HTTPException(
            status_code=500,
            detail="GitHub OAuth is not configured. Set GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET."
        )

    redirect_uri = request.url_for("auth_callback_github")

    return await oauth.github.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback/github")
async def auth_callback_github(request: Request):

    token = await oauth.github.authorize_access_token(request)
    access_token = token["access_token"]

    user = await upsert_user_from_github(access_token)

    return _issue_session_and_redirect(user["id"])


def _issue_session_and_redirect(user_id: str):

    jwt_token = create_access_token(user_id)

    response = RedirectResponse(url=FRONTEND_URL)

    response.set_cookie(
        key=COOKIE_NAME,
        value=jwt_token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 24 * 60 * 60,
    )

    return response


# =====================================================
# AUTH: ME / LOGOUT
# =====================================================

@app.get("/auth/me")
async def auth_me(user=Depends(get_optional_user)):

    if not user:
        return {"authenticated": False, "user": None}

    return {
        "authenticated": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "avatar_url": user["avatar_url"],
            "provider": user["provider"],
        }
    }


@app.post("/auth/logout")
async def auth_logout():

    resp = JSONResponse(content={"message": "Logged out"})
    resp.delete_cookie(COOKIE_NAME)

    return resp


# =====================================================
# NEW SESSION
# =====================================================

@app.post("/session/new")
async def new_session(user=Depends(get_current_user)):

    session_id = create_session(user["id"])

    return {

        "message": "Session created",

        "session_id": session_id

    }


# =====================================================
# LIST SESSIONS (chat history list)
# =====================================================

@app.get("/sessions")
async def list_sessions(user=Depends(get_current_user)):

    sessions = get_sessions(user["id"])

    return {

        "total_sessions": len(sessions),

        "sessions": sessions

    }


# =====================================================
# GET ONE SESSION'S MESSAGES (continue a previous chat)
# =====================================================

@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str, user=Depends(get_current_user)):

    owner = get_session_owner(session_id)

    if owner is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if owner != user["id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    return {

        "session_id": session_id,

        "messages": get_chat_history(session_id)

    }


# =====================================================
# DELETE A SESSION
# =====================================================

@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str, user=Depends(get_current_user)):

    owner = get_session_owner(session_id)

    if owner is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if owner != user["id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    delete_session(session_id)

    return {"message": "Session deleted", "session_id": session_id}


# =====================================================
# UPLOAD PDF
# =====================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user=Depends(get_current_user)):

    global faiss_index
    global pdf_chunks

    existing_files = {

        chunk["file_name"]

        for chunk in pdf_chunks

        if chunk.get("user_id") == user["id"]

    }

    if file.filename in existing_files:

        return {

            "message": "File already exists",

            "file_name": file.filename

        }

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    # namespace the file on disk per-user to avoid collisions between users
    safe_prefix = user["id"][:8]
    stored_filename = f"{safe_prefix}__{file.filename}"

    file_path = os.path.join(
        "uploads",
        stored_filename
    )

    with open(file_path, "wb") as buffer:

        buffer.write(
            await file.read()
        )

    text = extract_text_from_pdf(
        file_path
    )

    chunks = split_text_into_chunks(
        text
    )

    new_chunks = []

    for i, chunk in enumerate(chunks):

        new_chunks.append({

            "file_name": file.filename,

            "stored_filename": stored_filename,

            "user_id": user["id"],

            "page_number": chunk["page_number"],

            "chunk_number": i,

            "text": chunk["text"]

        })

    pdf_chunks.extend(new_chunks)

    embeddings = create_embeddings(

        [

            chunk["text"]

            for chunk in new_chunks

        ]

    )

    if faiss_index is None:

        faiss_index = create_faiss_index(
            embeddings
        )

    else:

        add_embeddings(
            faiss_index,
            embeddings
        )

    save_vector_store(
        faiss_index,
        pdf_chunks
    )

    print("=" * 60)
    print("UPLOAD SUCCESS")
    print("User:", user["id"])
    print("File:", file.filename)
    print("Chunks Added:", len(new_chunks))
    print("Total Chunks:", len(pdf_chunks))
    print("Total Vectors:", faiss_index.ntotal)
    print("=" * 60)

    return {

        "message": "PDF uploaded successfully",

        "file_name": file.filename,

        "chunks_added": len(new_chunks),

        "total_chunks": len(pdf_chunks),

        "total_vectors": faiss_index.ntotal

    }


# =====================================================
# DOCUMENT LIST (upload history) — scoped to current user
# =====================================================

@app.get("/documents")
async def get_documents(user=Depends(get_current_user)):

    global pdf_chunks

    documents = {}

    for chunk in pdf_chunks:

        if chunk.get("user_id") != user["id"]:
            continue

        name = chunk["file_name"]

        if name not in documents:

            documents[name] = {

                "file_name": name,

                "chunks": 0,

                "pages": set()

            }

        documents[name]["chunks"] += 1

        documents[name]["pages"].add(

            chunk["page_number"]

        )

    result = []

    for doc in documents.values():

        result.append({

            "file_name": doc["file_name"],

            "chunks": doc["chunks"],

            "pages": len(doc["pages"])

        })

    return {

        "total_documents": len(result),

        "documents": result

    }

# =====================================================
# EMOJI SUMMARY / CHEAT SHEET — shared helper
# =====================================================

def _get_user_document_text(file_name: str, user_id: str) -> str:
    """
    Collects a user's own chunks for a given file, in page/chunk order,
    and joins them into one text blob (capped so we don't blow past the
    LLM's context window on very large PDFs).
    """

    matching = [
        chunk for chunk in pdf_chunks
        if chunk["file_name"] == file_name and chunk.get("user_id") == user_id
    ]

    if not matching:
        return None

    matching.sort(key=lambda c: (c["page_number"], c["chunk_number"]))

    text = "\n\n".join(chunk["text"] for chunk in matching)

    # cap to roughly ~12k characters (~3k tokens) to stay well within
    # the model's context window even for large PDFs
    max_chars = 12000

    if len(text) > max_chars:
        text = text[:max_chars]

    return text


# =====================================================
# EMOJI SUMMARY
# =====================================================

@app.post("/documents/{file_name}/emoji-summary")
async def emoji_summary(file_name: str, user=Depends(get_current_user)):

    document_text = _get_user_document_text(file_name, user["id"])

    if document_text is None:
        raise HTTPException(status_code=404, detail="Document not found")

    summary = generate_emoji_summary(document_text)

    return {
        "file_name": file_name,
        "emoji_summary": summary
    }


# =====================================================
# ONE-CLICK CHEAT SHEET
# =====================================================

@app.post("/documents/{file_name}/cheat-sheet")
async def cheat_sheet(file_name: str, user=Depends(get_current_user)):

    document_text = _get_user_document_text(file_name, user["id"])

    if document_text is None:
        raise HTTPException(status_code=404, detail="Document not found")

    sheet = generate_cheat_sheet(document_text)

    return {
        "file_name": file_name,
        "cheat_sheet": sheet
    }


# =====================================================
# ASK
# =====================================================

@app.post("/ask")
async def ask(request: QuestionRequest, user=Depends(get_current_user)):

    global faiss_index
    global pdf_chunks

    owner = get_session_owner(request.session_id)

    if owner is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if owner != user["id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    user_has_docs = any(
        chunk.get("user_id") == user["id"]
        for chunk in pdf_chunks
    )

    if faiss_index is None or not user_has_docs:

        raise HTTPException(

            status_code=400,

            detail="Please upload a PDF first."

        )

    history = get_chat_history(
        request.session_id
    )

    response = generate_answer(

        request.question,

        faiss_index,

        pdf_chunks,

        request.session_id,

        history,

        user["id"]

    )

    return response


# =====================================================
# STREAMING
# =====================================================

@app.post("/ask-stream")
async def ask_stream(request: QuestionRequest, user=Depends(get_current_user)):

    global faiss_index
    global pdf_chunks

    owner = get_session_owner(request.session_id)

    if owner is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if owner != user["id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    user_has_docs = any(
        chunk.get("user_id") == user["id"]
        for chunk in pdf_chunks
    )

    if faiss_index is None or not user_has_docs:

        raise HTTPException(

            status_code=400,

            detail="Please upload a PDF first."

        )

    history = get_chat_history(
        request.session_id
    )

    generator = stream_answer(

        request.question,

        faiss_index,

        pdf_chunks,

        request.session_id,

        history,

        user["id"]

    )

    return StreamingResponse(

        generator,

        media_type="text/event-stream"

    )


# =====================================================
# DELETE DOCUMENT
# =====================================================

@app.delete("/documents/{file_name}")
async def delete_document(file_name: str, user=Depends(get_current_user)):

    global pdf_chunks
    global faiss_index

    existing_chunks = [

        chunk

        for chunk in pdf_chunks

        if chunk["file_name"] == file_name and chunk.get("user_id") == user["id"]

    ]

    if len(existing_chunks) == 0:

        raise HTTPException(

            status_code=404,

            detail="Document not found"

        )

    # -----------------------------------
    # Delete PDF from uploads folder
    # -----------------------------------

    stored_filename = existing_chunks[0].get("stored_filename", file_name)

    file_path = os.path.join(

        "uploads",

        stored_filename

    )

    if os.path.exists(file_path):

        os.remove(file_path)

        print("Deleted PDF:", stored_filename)

    # -----------------------------------
    # Remove chunks (only this user's copy of this file)
    # -----------------------------------

    pdf_chunks = [

        chunk

        for chunk in pdf_chunks

        if not (chunk["file_name"] == file_name and chunk.get("user_id") == user["id"])

    ]

    print("Remaining Chunks:", len(pdf_chunks))

    # -----------------------------------
    # No documents left at all
    # -----------------------------------

    if len(pdf_chunks) == 0:

        faiss_index = None

        save_vector_store(

            faiss_index,

            pdf_chunks

        )

        print("Vector Store Cleared")

        return {

            "message": "All documents deleted",

            "remaining_chunks": 0,

            "faiss_vectors": 0

        }

    # -----------------------------------
    # Rebuild FAISS over everyone's remaining chunks
    # (index positions must stay aligned with pdf_chunks)
    # -----------------------------------

    texts = [

        chunk["text"]

        for chunk in pdf_chunks

    ]

    embeddings = create_embeddings(

        texts

    )

    faiss_index = create_faiss_index(

        embeddings

    )

    save_vector_store(

        faiss_index,

        pdf_chunks

    )

    print("=" * 60)
    print("DOCUMENT DELETED")
    print("User:", user["id"])
    print("Remaining Chunks :", len(pdf_chunks))
    print("Remaining Vectors:", faiss_index.ntotal)
    print("=" * 60)

    return {

        "message": f"{file_name} deleted successfully",

        "remaining_chunks": len(pdf_chunks),

        "faiss_vectors": faiss_index.ntotal

    }

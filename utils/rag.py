from utils.embedding import create_embedding
from utils.vector_store import search_faiss, search_faiss_scoped
from utils.hybrid_search import bm25_search
import json

from utils.llm import (
    ask_groq_with_memory,
    rewrite_question,
    generate_chat_title,
    stream_groq
)

from utils.memory import (
    add_message,
    update_session_title
)



##############################################################
# HYBRID RETRIEVAL
##############################################################

def retrieve_context(
        search_question,
        query_embedding,
        faiss_index,
        pdf_chunks,
        user_id=None
):

    """
    Hybrid Search:
    FAISS (semantic)
    +
    BM25 (keyword)

    The FAISS index and pdf_chunks list are shared across all users. We
    filter down to the requesting user's own chunks FIRST and then run
    FAISS + BM25 only within that subset — rather than searching the
    whole shared store and filtering afterwards. Searching first and
    filtering after can silently drop a user's own best-matching chunk
    if it doesn't happen to land in the global top-k, especially once
    the shared store has a lot of other content in it.
    """

    k = 15



    # ---------------- FAISS SEARCH ----------------

    if user_id is not None:
        faiss_sources = search_faiss_scoped(
            faiss_index,
            query_embedding,
            pdf_chunks,
            user_id,
            k=k
        )
    else:
        faiss_sources = search_faiss(
            faiss_index,
            query_embedding,
            pdf_chunks,
            k=k
        )



    # ---------------- BM25 SEARCH ----------------

    bm25_sources = bm25_search(

        search_question,

        pdf_chunks,

        k=k,

        user_id=user_id

    )



    # ---------------- MERGE RESULTS ----------------


    merged = []

    seen = set()



    for source in faiss_sources + bm25_sources:


        key = (

            source["file_name"],

            source["chunk_number"]

        )


        if key not in seen:


            seen.add(key)

            merged.append(source)



    # take best 15 chunks

    sources = merged[:15]




    # ---------------- DEBUG ----------------


    print("\n" + "="*60)

    print("FAISS RESULTS :",len(faiss_sources))

    print("BM25 RESULTS  :",len(bm25_sources))

    print("FINAL CHUNKS  :",len(sources))

    print("="*60)



    for s in sources:

        print(

            s["file_name"],

            "| Page:",

            s["page_number"],

            "| Chunk:",

            s["chunk_number"]

        )



    # ---------------- CONTEXT BUILDING ----------------


    context_parts=[]


    for index,source in enumerate(sources):


        context_parts.append(

f"""
SOURCE {index+1}

File:
{source['file_name']}

Page:
{source['page_number']}


CONTENT:

{source['text']}

"""

        )



    context="\n\n".join(context_parts)



    print(
        "Context characters:",
        len(context)
    )



    return context,sources





##############################################################
# NORMAL ANSWER
##############################################################

def generate_answer(

        question,

        faiss_index,

        pdf_chunks,

        session_id,

        history,

        user_id=None

):



    # Rewrite question for memory search

    if history:


        search_question = rewrite_question(

            question,

            history

        )


    else:

        search_question = question




    print("\nQuestion:",question)

    print("Search query:",search_question)




    # Create title first chat

    if not history:


        title = generate_chat_title(

            question

        )


        update_session_title(

            session_id,

            title

        )





    query_embedding = create_embedding(

        search_question

    )





    context,sources = retrieve_context(

        search_question,

        query_embedding,

        faiss_index,

        pdf_chunks,

        user_id

    )





    answer = ask_groq_with_memory(

        question,

        context,

        history

    )





    add_message(

        session_id,

        "user",

        question

    )



    add_message(

        session_id,

        "assistant",

        answer

    )





    return {


        "question":question,


        "answer":answer,


        "sources":[


            {

            "file_name":s["file_name"],

            "page_number":s["page_number"],

            "chunk_number":s["chunk_number"],

            "text":s["text"]

            }


            for s in sources


        ]

    }





##############################################################
# STREAM ANSWER
##############################################################

##############################################################
# STREAM ANSWER
##############################################################

def stream_answer(

        question,

        faiss_index,

        pdf_chunks,

        session_id,

        history,

        user_id=None

):


    if history:

        search_question = rewrite_question(
            question,
            history
        )

    else:

        search_question = question





    # Generate title for first chat

    if not history:

        title = generate_chat_title(
            question
        )

        update_session_title(
            session_id,
            title
        )





    # Create embedding

    query_embedding = create_embedding(
        search_question
    )





    # Retrieve context

    context, sources = retrieve_context(

        search_question,

        query_embedding,

        faiss_index,

        pdf_chunks,

        user_id

    )





    # Save user message

    add_message(

        session_id,

        "user",

        question

    )





    full_answer = ""





    # Send answer tokens

    for token in stream_groq(

        question,

        context,

        history

    ):


        full_answer += token



        yield json.dumps({

            "type":"token",

            "content":token

        }) + "\n"







    # Send sources after answer complete

    yield json.dumps({

        "type":"sources",

        "sources":[

            {

                "file_name":s["file_name"],

                "page_number":s["page_number"],

                "chunk_number":s["chunk_number"],

                "text":s["text"]

            }

            for s in sources

        ]

    }) + "\n"






    # Save assistant message

    add_message(

        session_id,

        "assistant",

        full_answer

    )
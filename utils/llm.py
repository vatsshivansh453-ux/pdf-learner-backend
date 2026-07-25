from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


MODEL_NAME = "llama-3.3-70b-versatile"



##############################################################
# COMMON SYSTEM PROMPT
##############################################################

SYSTEM_PROMPT = """

You are an expert AI PDF assistant having a normal conversation with the user.

Your job is to answer questions using ONLY the provided PDF context, and to
remember and use the earlier turns of the conversation (given to you below)
so you can correctly resolve follow-up questions like "explain its types" or
"what about the second one" without the user having to repeat themselves.

Answer style — write like a knowledgeable person chatting, NOT like a
formatted document:

- Default to plain, normal paragraphs, just like a regular chat reply.
- Do NOT force every answer into headings/sections. Only add a heading if
  the answer is genuinely long and multi-part enough to need one.
- Only use a bullet or numbered list when the content is actually a list
  (steps, types, categories). Do not convert plain explanations into lists.
- Use **bold** sparingly, only for the 1-2 most important terms — not for
  every sentence or every heading.
- Keep answers as short as they can be while still being complete. A simple
  question deserves a simple, short answer (a sentence or a short paragraph),
  not a template with "Definition / Key Points / Summary" sections.
- No emojis.
- Never invent information that isn't in the PDF context.

If the information is not available in the PDF, say:

"I couldn't find that information in the uploaded PDF."

"""



##############################################################
# NORMAL GROQ ANSWER
##############################################################

def ask_groq(question, context):


    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },

            {
                "role":"user",
                "content":f"""

PDF Context:

{context}


Question:

{question}

"""
            }

        ],

        temperature=0,

        max_tokens=2000

    )


    return response.choices[0].message.content





##############################################################
# GROQ ANSWER WITH MEMORY
##############################################################

def ask_groq_with_memory(
        question,
        context,
        history
):


    history_text = ""


    for message in history:

        history_text += (

            message["role"]

            +

            ": "

            +

            message["content"]

            +

            "\n"

        )



    prompt = f"""

Conversation History:

{history_text}


PDF Context:

{context}


Current Question:

{question}


Answer using Markdown format.

"""



    response = client.chat.completions.create(

        model=MODEL_NAME,


        messages=[

            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },

            {
                "role":"user",
                "content":prompt
            }

        ],


        temperature=0,

        max_tokens=2000

    )



    return response.choices[0].message.content





##############################################################
# QUESTION REWRITE
##############################################################

def rewrite_question(question, history):


    history_text = ""


    for message in history:

        history_text += (

            message["role"]

            +

            ": "

            +

            message["content"]

            +

            "\n"

        )



    prompt = f"""

Conversation:

{history_text}


Current Question:

{question}


Rewrite this into a complete standalone question.

Only return the rewritten question.

"""



    response = client.chat.completions.create(

        model=MODEL_NAME,


        messages=[

            {
                "role":"user",
                "content":prompt
            }

        ],


        temperature=0

    )



    return response.choices[0].message.content





##############################################################
# CHAT TITLE GENERATION
##############################################################

def generate_chat_title(question):


    response = client.chat.completions.create(

        model=MODEL_NAME,


        messages=[

            {

                "role":"system",

                "content":"""

Generate a short chat title.

Rules:

- Maximum 5 words.
- No punctuation.
- No quotes.
- Return only title.

"""

            },


            {

                "role":"user",

                "content":question

            }

        ],


        temperature=0

    )



    return response.choices[0].message.content.strip()





##############################################################
# EMOJI SUMMARY
##############################################################

def generate_emoji_summary(document_text):


    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role":"system",
                "content":"""

You are summarizing a PDF document as a fun, skimmable "emoji summary".

Rules:

- Produce 6 to 10 lines.
- Each line starts with ONE relevant emoji, then a short phrase (max 12 words) capturing one key idea from the document.
- Cover the most important ideas in the document, in the order they appear.
- No headings, no numbering, no extra commentary — only the emoji lines.
- Keep phrases concrete and specific to THIS document, never generic.

Example format:

📌 Explains how photosynthesis converts light into energy
🌱 Chlorophyll absorbs sunlight in the leaves
💧 Water and CO2 are the raw materials
☀️ Light reactions happen in the thylakoid membrane

"""
            },

            {
                "role":"user",
                "content":f"Document:\n\n{document_text}"
            }

        ],

        temperature=0.3,

        max_tokens=500

    )


    return response.choices[0].message.content.strip()




##############################################################
# ONE-CLICK CHEAT SHEET
##############################################################

def generate_cheat_sheet(document_text):


    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role":"system",
                "content":"""

You create a compact "cheat sheet" for a PDF document — the kind of thing a student
would print out before an exam.

Rules:

- Use Markdown.
- Start with a `Cheat Sheet` heading.
- Include a `Key Terms` section: term — one-line definition, for every important term in the document.
- Include a `Must-Know Points` section: the most important facts/points as a tight bullet list.
- Include a `Quick Formulas / Rules` section ONLY if the document contains formulas, equations, or rules — otherwise omit this section entirely.
- Be dense and skimmable. No long paragraphs. No filler sentences.
- Only use information found in the document. Do not invent anything.
-create a pdf file of conatining the text you generated for the user and give to user 

"""
            },

            {
                "role":"user",
                "content":f"Document:\n\n{document_text}"
            }

        ],

        temperature=0,

        max_tokens=1500

    )


    return response.choices[0].message.content.strip()




##############################################################
# STREAMING RESPONSE
##############################################################

def stream_groq(question, context, history=None):

    history = history or []

    history_text = ""

    for message in history:

        history_text += (
            message["role"]
            +
            ": "
            +
            message["content"]
            +
            "\n"
        )

    prompt = f"""

Conversation History:

{history_text if history_text else "(no earlier messages)"}


PDF Context:

{context}


Current Question:

{question}


Answer the Current Question, using the Conversation History to understand
what any pronouns or references ("it", "its", "that", "the second one", etc.)
in the Current Question refer to.

"""

    stream = client.chat.completions.create(

        model=MODEL_NAME,


        messages=[


            {

                "role":"system",

                "content":SYSTEM_PROMPT

            },


            {

                "role":"user",

                "content":prompt

            }


        ],


        temperature=0,


        max_tokens=2000,


        stream=True

    )



    for chunk in stream:


        if chunk.choices[0].delta.content:


            yield chunk.choices[0].delta.content
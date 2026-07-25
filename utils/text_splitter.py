from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    chunks=[]


    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )


        for chunk in page_chunks:

            chunks.append(
                {
                    "page_number": page["page_number"],
                    "text": chunk
                }
            )


    return chunks
from pypdf import PdfReader

pdf_path = "uploads/Introduction_to_AI_One_Page.pdf"

reader=PdfReader(pdf_path)
text=""
for page in reader.pages:
    text+=page.extract_text()
print(text)
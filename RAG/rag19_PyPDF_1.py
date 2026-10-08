# PDF 논문을 페이지별 문서로 불러오기

# pip install pypdf

from langchain_community.document_loaders import TextLoader, PyPDFLoader


path = "./_data/"
pdf_loader = PyPDFLoader(path + "1706.03762v7.pdf")

pdf_docs = pdf_loader.load()
print(type(pdf_docs)) # <class 'list'>
print(len(pdf_docs)) # 15 # 페이지
print(pdf_docs)

print("=====================================")
print(pdf_docs[0])
print("=====================================")

###############################################################################
# PDF 전체를 하나의 Document로 불러오기


pdf_loader = PyPDFLoader(path + "1706.03762v7.pdf", mode="single")

pdf_docs = pdf_loader.load()
print(type(pdf_docs)) # <class 'list'>
print(len(pdf_docs)) # 1 # 페이지
print(pdf_docs)

print("=====================================")
print(pdf_docs[0])
print("=====================================")

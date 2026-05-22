import requests
from bs4 import BeautifulSoup as bs

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import numpy as np

def single_page_scrapper(url):
    try:
        response = requests.get(url)
        soup = bs(response.text, 'html.parser')
    except:
        return None
    return soup

#word tokenizer
def tokenizeW(txt):
    doc = nlp(txt)
    tkn=[]
    for token in doc:
         if token.is_alpha:
             tkn.append(token.text)
    return tkn

#sentence tokenizer
def tokenizeS(txt):
    doc = nlp(txt)
    tkn=[]
    for token in doc.sents:
            tkn.append(token.text.strip())
    return tkn

def sentEmbed(tkn):
    embeddings = model.encode(tkn)
    return embeddings





txt=''

#later create a loop to scrap multiple pages from the links in homepage
k=single_page_scrapper("https://lite.cnn.com/2026/05/21/tech/south-korea-samsung-strike-intl-hnk")
paragraphs= k.find_all('p', class_='paragraph--lite')


for p in paragraphs:
    if p.text!="See Full Web Article":
        txt+=p.text
# with open("news.txt","w") as txtFile:
#     txtFile.write(txt)


tkn=tokenizeS(txt)
# print(tkn)

embed=sentEmbed(tkn)
#np.save("embed.npy", embed)

print(embed)
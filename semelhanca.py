import spacy
nlp = spacy.load("pt_core_news_md")
#nlp = spacy.load("en_core_web_md")


doc1 = nlp(u'Eu sou o seu pai')
doc2 = nlp(u'Sou seu pai')
doc3 = nlp(u'Eu sou o seu tio')


print(doc1.similarity(doc2)) 
print(doc1.similarity(doc3))
print(doc2.similarity(doc3))
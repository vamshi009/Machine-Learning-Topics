import spacy

def entites(text, lang_code):    
    if(lang_code=="en"):
        model = spacy.load("xx_ent_wiki_sm")
    else:
        try:
            model = spacy.load(lang_code + '_core_news_lg')
        except:
            model = spacy.load("xx_ent_wiki_sm")

    anslist = []
    for x in model(str(text)).ents:
        dik = {}
        dik['text'] = str(x)
        dik['type'] = x.label_
        dik['start_pos'] = x.start_char
        dik['end_pos'] = x.end_char
        anslist.append(dik)

    return anslist


if(__name__=="__main__"):
    print(entites("I love India, Europe, USA, Hyderabad, Sachin and Michal Jackson", "en"))
    print(entites("Apple overweegt om voor 1 miljard een U.K. startup te kopen", "nl"))


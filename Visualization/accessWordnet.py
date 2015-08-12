__author__ = 'mcapizzi'

from nltk.corpus import wordnet

# http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

def get_wn_relation(word, relationship):
    worddict = []

    # synonyms (approximation using synsets)
    if relationship == "s":
        for item in wordnet.synsets(word):
            for l in item.lemmas():
                if l.name() != "good":
                    worddict.append(l.name())
                else:
                    continue
        return worddict

    # antonyms
    elif relationship == "a":
        for item in wordnet.synsets(word):
            for l in item.lemmas():
                for a in l.antonyms():
                    if a.name() not in worddict:
                        worddict.append(a.name())
                    else:
                        continue
        return worddict

    # hypernyms
    elif relationship == "hr":
        for item in wordnet.synsets(word):
            for hr in item.hypernyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

    # hyponyms
    elif relationship == "ho":
        for item in wordnet.synsets(word):
            for hr in item.hyponyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

    # member_holonyms
    elif relationship == "mh":
        for item in wordnet.synsets(word):
            for hr in item.member_holonyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

    # part_holonyms
    elif relationship == "ph":
        for item in wordnet.synsets(word):
            for hr in item.part_holonyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

    # member_meronyms
    elif relationship == "mm":
        for item in wordnet.synsets(word):
            for hr in item.member_meronyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

    # part_meronyms
    elif relationship == "pm":
        for item in wordnet.synsets(word):
            for hr in item.part_meronyms():
                for l in hr.lemmas():
                    if l.name() not in worddict:
                        worddict.append(l.name())
                    else:
                        continue
        return worddict

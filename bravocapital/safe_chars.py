# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:16:03 2015

@author: mgadi
"""

import string
 
def normalize(s):
    for p in string.punctuation:
        s = s.replace(p, '')
    return s.upper().strip()
    
import re
legalchars = "abcdefghijklmnopqrstuvwxyx1234567890"
replace = (
    (("Ã","Å","Ä","À","Á","Â","å","å","ä","à","á","â"),"a"),
    (("Ç","Č","ç","č"),"c"),
    (("É","È","Ê","Ë","Ĕ","è","ê","ë","ĕ","é"),"e"),
    (("Ğ","Ģ","ģ","ğ"),"g"),
    (("Ï","Î","Í","Ì","ï","î","í","ì"),"i"),
    (("Ñ","ñ"),"n"),
    (("Ö","Ô","Ō","Ò","Ó","Ø","ö","ô","ō","ò","ó","ø"),"o"),
    (("Ŝ","Ş","ŝ","ş"),"s"),
    (("Ü","Ū","Û","Ù","Ú","ü","ū","û","ù","ú"),"u"),
    (("Ÿ","ÿ","&"),"y"),
    (("@")," ")
    )

def remove_chars(subject):
    """ Replace all chars that arent in allowed list """
    return re.sub(r'[^\x00-\x7f]',r' ',subject).replace("nan", " ")
    
def safe_chars_nombrepersona(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    a = remove_chars(subject)
    return a.title().replace('Do A ','').replace('Dona ','').replace('Don ','').replace('Mr ','').replace('Mrs ','').replace('Ms ','')

def safe_upperstr(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    s = remove_chars(normalize(subject))
    return s.upper()

def safe_upperstr_razonsocial(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    s = remove_chars(normalize(subject))
    return s.upper().replace('.','').replace(' SL','').replace(' SA','').replace(' SAU','').replace(' SLU','').replace(' SLP','').replace(' SLL','').replace(' SC','')

    
#print safe_chars("España &@")

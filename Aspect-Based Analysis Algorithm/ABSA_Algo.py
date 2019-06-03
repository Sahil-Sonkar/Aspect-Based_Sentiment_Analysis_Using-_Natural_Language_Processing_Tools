# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:51:40 2019

@author: Sahil Sonkar
"""
import spacy
nlp_en= spacy.load('en_core_web_sm')

from enum import Enum

class Topic(Enum):
    AMBIENCE = 1
    FOOD = 2
    SERVICE = 3
    VALUE = 4
    
class Rating(Enum):
    VERY_BAD = -3
    BAD = -2
    SOMEWHAT_BAD = -1
    SOMEWHAT_GOOD = 1
    GOOD = 2
    VERY_GOOD = 3
    
import re
from spacy.tokens import Token

class LexiconEntry:
    _IS_REGEX_REGEX = re.compile(r'.*[.+*\[$^\\]')

    def __init__(self, lemma: str, topic: Topic, rating: Rating):
        assert lemma is not None
        self.lemma = lemma
        self._lower_lemma = lemma.lower()
        self.topic = topic
        self.rating = rating
        self.is_regex = bool(LexiconEntry._IS_REGEX_REGEX.match(self.lemma))
        self._regex = re.compile(lemma, re.IGNORECASE) if self.is_regex else None

    def matching(self, token: Token) -> float:
        """
        A weight between 0.0 and 1.0 on how much ``token`` matches this entry.
        """
        assert token is not None
        result = 0.0
        if self.is_regex:
            if self._regex.match(token.text):
                result = 0.6
            elif self._regex.match(token.lemma_):
                result = 0.5
        else:
            if token.text == self.lemma:
                result = 1.0
            elif token.text.lower() == self.lemma:
                result = 0.9
            elif token.lemma_ == self.lemma:
                result = 0.8
            elif token.lemma_.lower() == self.lemma:
                result = 0.7
        return result

    def __str__(self) -> str:
        result = 'LexiconEntry(%s' % self.lemma
        if self.topic is not None:
            result += ', topic=%s' % self.topic.name
        if self.rating is not None:
            result += ', rating=%s' % self.rating.name
        if self.is_regex:
            result += ', is_regex=%s' % self.is_regex
        result += ')'
        return result

    def __repr__(self) -> str:
        return self.__str__()
    
from math import isclose

class Lexicon:
    def __init__(self):
        self.entries: List[LexiconEntry] = []

    
    def append(self, lemma: str, topic: Topic, rating: Rating):
        lexicon_entry = LexiconEntry(lemma, topic, rating)
        self.entries.append(lexicon_entry)

    def lexicon_entry_for(self, token: Token) -> LexiconEntry:
        """
        Entry in lexicon that best matches ``token``.
        """
        result = None
        lexicon_size = len(self.entries)
        lexicon_entry_index = 0
        best_matching = 0.0
        while lexicon_entry_index < lexicon_size and not isclose(best_matching, 1.0):
            lexicon_entry = self.entries[lexicon_entry_index]
            matching = lexicon_entry.matching(token)
            if matching > best_matching:
                result = lexicon_entry
                best_matching = matching
            lexicon_entry_index += 1
        return result
    
import csv   
import pandas as pd
  
lexicon = Lexicon()
#lexi = pd.read_csv('lexicon.csv', delimiter = ',')

#print(lexi.iloc[0,1])

lexicon.append('waiter'     , Topic.SERVICE , None)
lexicon.append('waitress'   , Topic.SERVICE , None)
lexicon.append('wait'       , None          , Rating.BAD)
lexicon.append('quick'      , None          , Rating.GOOD)
lexicon.append('.*schnitzel', Topic.FOOD    , None)
lexicon.append('music'      , Topic.AMBIENCE, None)
lexicon.append('loud'       , None          , Rating.BAD)
lexicon.append('tasty'      , Topic.FOOD    , Rating.GOOD)
lexicon.append('polite'     , Topic.SERVICE , Rating.GOOD)
lexicon.append('wow',None,Rating.VERY_GOOD)
lexicon.append('loved',None,Rating.VERY_GOOD)
lexicon.append('bad',None,Rating.BAD)
lexicon.append('good',None,Rating.GOOD)
lexicon.append('nasty',None,Rating.SOMEWHAT_BAD)
lexicon.append('texture',Topic.FOOD,None)
lexicon.append('great',None,Rating.VERY_GOOD)
lexicon.append('treat',Topic.FOOD,Rating.VERY_GOOD)
lexicon.append('prices',Topic.VALUE,None)
lexicon.append('place',Topic.AMBIENCE,None)
lexicon.append('place',Topic.AMBIENCE,None)
lexicon.append('.*crust',Topic.FOOD,None)
lexicon.append('angry',None,Rating.VERY_BAD)
lexicon.append('damn',None,Rating.SOMEWHAT_BAD)
lexicon.append('like',None,Rating.GOOD)
lexicon.append('service',Topic.SERVICE,None)
lexicon.append('food',Topic.FOOD,None)
lexicon.append('ambience',Topic.AMBIENCE,None)
lexicon.append('value',Topic.VALUE,None)
lexicon.append('disgusted',None,Rating.VERY_BAD)
lexicon.append('disgust',None,Rating.VERY_BAD)
lexicon.append('disgusting',None,Rating.VERY_BAD)
lexicon.append('dissapointed',None,Rating.VERY_BAD)
lexicon.append('dissapoint',None,Rating.VERY_BAD)
lexicon.append('dissapointing',None,Rating.VERY_BAD)
lexicon.append('fresh',Topic.FOOD,Rating.GOOD)
lexicon.append('driest',Topic.FOOD,Rating.BAD)
lexicon.append('warm',Topic.FOOD,Rating.GOOD)
lexicon.append('good',None,Rating.GOOD)
lexicon.append('overpriced',Topic.VALUE,Rating.BAD)
lexicon.append('expensive',Topic.VALUE,Rating.BAD)
lexicon.append('inexpensive',Topic.VALUE,Rating.GOOD)
lexicon.append('shocked',None,Rating.BAD)
lexicon.append('cash',Topic.VALUE,None)
lexicon.append('recommended',None,Rating.GOOD)
lexicon.append('recommend',None,Rating.GOOD)
lexicon.append('slow',Topic.SERVICE,Rating.BAD)
lexicon.append('people',Topic.SERVICE,None)
lexicon.append('jerk',Topic.SERVICE,Rating.BAD)
lexicon.append('rushed',Topic.SERVICE,Rating.BAD)
lexicon.append('insulted',Topic.SERVICE,Rating.BAD)
lexicon.append('forgetting',Topic.SERVICE,Rating.BAD)
lexicon.append('rude',Topic.SERVICE,Rating.BAD)
lexicon.append('rudely',Topic.SERVICE,Rating.BAD)
lexicon.append('unsatisfying',Topic.SERVICE,Rating.BAD)
lexicon.append('inconsistent',Topic.SERVICE,Rating.BAD)
lexicon.append('inconsiderate',Topic.SERVICE,Rating.BAD)
lexicon.append('busy',Topic.SERVICE,Rating.BAD)
lexicon.append('worth',None,Rating.GOOD)
lexicon.append('amazing',None,Rating.VERY_GOOD)
lexicon.append('interesting',None,Rating.VERY_GOOD)
lexicon.append('cute',None,Rating.SOMEWHAT_GOOD)
lexicon.append('beautiful',None,Rating.GOOD)
lexicon.append('beautifullly',None,Rating.GOOD)
lexicon.append('interior',Topic.AMBIENCE,None)
lexicon.append('ventilation',Topic.AMBIENCE,None)
lexicon.append('clean',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('spot',Topic.AMBIENCE,None)
lexicon.append('tiny',Topic.AMBIENCE,Rating.BAD)
lexicon.append('perfectly',None,Rating.GOOD)
lexicon.append('better',None,Rating.VERY_GOOD)
lexicon.append('perfect',None,Rating.GOOD)
lexicon.append('back',None,Rating.GOOD)
lexicon.append('perfection',None,Rating.GOOD)
lexicon.append('authentic',None,Rating.GOOD)
lexicon.append('cooked',Topic.FOOD,None)
lexicon.append('.*potatoes',Topic.FOOD,None)
lexicon.append('.*fries',Topic.FOOD,None)
lexicon.append('.*rolls',Topic.FOOD,None)
lexicon.append('.*chicken',Topic.FOOD,None)
lexicon.append('.*cranberry',Topic.FOOD,None)
lexicon.append('.*ravoli',Topic.FOOD,None)
lexicon.append('.*hair',Topic.FOOD,Rating.VERY_BAD)
lexicon.append('.*poop',Topic.FOOD,Rating.VERY_BAD)
lexicon.append('.*cake',Topic.FOOD,None)
lexicon.append('.*salad',Topic.FOOD,None)
lexicon.append('.*tacos',Topic.FOOD,None)
lexicon.append('.*sushi',Topic.FOOD,None)
lexicon.append('.*burger',Topic.FOOD,None)
lexicon.append('.*beer',Topic.FOOD,None)
lexicon.append('.*duck',Topic.FOOD,None)
lexicon.append('.*sauce',Topic.FOOD,None)
lexicon.append('.*crawfish',Topic.FOOD,None)
lexicon.append('.*cocktails',Topic.FOOD,None)
lexicon.append('.*pancake',Topic.FOOD,None)
lexicon.append('.*pasta',Topic.FOOD,None)
lexicon.append('.*steaks',Topic.FOOD,None)
lexicon.append('.*shrimp',Topic.FOOD,None)
lexicon.append('.*crab',Topic.FOOD,None)
lexicon.append('.*tartare',Topic.FOOD,None)
lexicon.append('.*wagyu',Topic.FOOD,None)
lexicon.append('.*cheese',Topic.FOOD,None)
lexicon.append('.*pizza',Topic.FOOD,None)
lexicon.append('.*chowmein',Topic.FOOD,None)
lexicon.append('.*sandwich',Topic.FOOD,None)
lexicon.append('.*margaritas',Topic.FOOD,None)
lexicon.append('.*tea',Topic.FOOD,None)
lexicon.append('.*rice',Topic.FOOD,None)
lexicon.append('.*beans',Topic.FOOD,None)
lexicon.append('.*eggplant',Topic.FOOD,None)
lexicon.append('.*pears',Topic.FOOD,None)
lexicon.append('.*pita',Topic.FOOD,None)
lexicon.append('.*hummus',Topic.FOOD,None)
lexicon.append('.*almond',Topic.FOOD,None)
lexicon.append('.*bacon',Topic.FOOD,None)
lexicon.append('.*biscuits',Topic.FOOD,None)
lexicon.append('.*omelets',Topic.FOOD,None)
lexicon.append('.*garlic',Topic.FOOD,None)
lexicon.append('.*marrow',Topic.FOOD,None)
lexicon.append('.*seafood',Topic.FOOD,None)
lexicon.append('.*fish',Topic.FOOD,None)
lexicon.append('.*beef',Topic.FOOD,None)
lexicon.append('.*milk',Topic.FOOD,None)
lexicon.append('.*milkshake',Topic.FOOD,None)
lexicon.append('.*banana',Topic.FOOD,None)
lexicon.append('.*rib',Topic.FOOD,None)
lexicon.append('.*sashimi',Topic.FOOD,None)
lexicon.append('appetizer',Topic.FOOD,None)
lexicon.append('eat',Topic.FOOD,None)
lexicon.append('eating',Topic.FOOD,None)
lexicon.append('tender',Topic.FOOD,Rating.GOOD)
lexicon.append('brunch',Topic.FOOD,None)
lexicon.append('breakfast',Topic.FOOD,None)
lexicon.append('lunch',Topic.FOOD,None)
lexicon.append('dinner',Topic.FOOD,None)
lexicon.append('buffet',Topic.FOOD,None)
lexicon.append('buffets',Topic.FOOD,None)
lexicon.append('gourmet',Topic.FOOD,None)
lexicon.append('dessert',Topic.FOOD,None)
lexicon.append('flavor',Topic.FOOD,None)
lexicon.append('spicy',Topic.FOOD,None)
lexicon.append('yummy',Topic.FOOD,Rating.VERY_GOOD)
lexicon.append('tasteless',Topic.FOOD,Rating.VERY_BAD)
lexicon.append('friendly',Topic.SERVICE,Rating.GOOD)
lexicon.append('familiar',Topic.SERVICE,Rating.GOOD)
lexicon.append('enthusiastic',Topic.SERVICE,Rating.GOOD)
lexicon.append('attention',Topic.SERVICE,Rating.GOOD)
lexicon.append('attitudes',Topic.SERVICE,Rating.BAD)
lexicon.append('helpful',Topic.SERVICE,Rating.GOOD)
lexicon.append('server',Topic.SERVICE,None)
lexicon.append('worst',None,Rating.VERY_BAD)
lexicon.append('petrified',None,Rating.VERY_BAD)
lexicon.append('disaster',None,Rating.VERY_BAD)
lexicon.append('old',None,Rating.VERY_BAD)
lexicon.append('mediocre',None,Rating.SOMEWHAT_BAD)
lexicon.append('upgrading',None,Rating.BAD)
lexicon.append('avoid',None,Rating.BAD)
lexicon.append('decorated',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('comfortable',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('letdown',None,Rating.VERY_BAD)
lexicon.append('awkward',None,Rating.VERY_BAD)
lexicon.append('waste',None,Rating.VERY_BAD)
lexicon.append('generic',None,Rating.BAD)
lexicon.append('decent',None,Rating.SOMEWHAT_GOOD)
lexicon.append('enjoy',None,Rating.GOOD)
lexicon.append('enjoyed',None,Rating.GOOD)
lexicon.append('phenomenal',None,Rating.GOOD)
lexicon.append('excellent',None,Rating.GOOD)
lexicon.append('incredible',None,Rating.GOOD)
lexicon.append('imaginative',None,Rating.GOOD)
lexicon.append('excuse',None,Rating.BAD)
lexicon.append('happy',None,Rating.GOOD)
lexicon.append('fantastic',None,Rating.VERY_GOOD)
lexicon.append('special',None,Rating.VERY_GOOD)
lexicon.append('happier',None,Rating.VERY_GOOD)
lexicon.append('happiest',None,Rating.VERY_GOOD)
lexicon.append('wonderful',None,Rating.VERY_GOOD)
lexicon.append('elsewhere',None,Rating.BAD)
lexicon.append('meh',None,Rating.BAD)
lexicon.append('terrible',None,Rating.BAD)
lexicon.append('underwhelming',None,Rating.BAD)
lexicon.append('appalling',None,Rating.BAD)
lexicon.append('bland',Topic.FOOD,Rating.BAD)
lexicon.append('blandest',Topic.FOOD,Rating.BAD)
lexicon.append('disgrace',None,Rating.BAD)
lexicon.append('overwhelmed',None,Rating.BAD)
lexicon.append('favourite',None,Rating.GOOD)
lexicon.append('winner',None,Rating.GOOD)
lexicon.append('jewel',None,Rating.GOOD)
lexicon.append('delight',None,Rating.GOOD)
lexicon.append('tragedy',None,Rating.BAD)
lexicon.append('frustrated',None,Rating.BAD)
lexicon.append('drag',None,Rating.BAD)
lexicon.append('poor',None,Rating.BAD)
lexicon.append('full',Topic.FOOD,Rating.GOOD)
lexicon.append('hit',None,Rating.GOOD)
lexicon.append('filling',Topic.FOOD,Rating.GOOD)
lexicon.append('stuffed',None,Rating.GOOD)
lexicon.append('diverse',Topic.FOOD,Rating.GOOD)
lexicon.append('horrible',None,Rating.VERY_BAD)
lexicon.append('humiliated',None,Rating.VERY_BAD)
lexicon.append('sucks',None,Rating.VERY_BAD)
lexicon.append('screwed',None,Rating.VERY_BAD)
lexicon.append('sad',None,Rating.VERY_BAD)
lexicon.append('stale',Topic.FOOD,Rating.VERY_BAD)
lexicon.append('awesome',None,Rating.VERY_GOOD)
lexicon.append('terrific',None,Rating.VERY_GOOD)
lexicon.append('thrilled',None,Rating.VERY_GOOD)
lexicon.append('impeccable',None,Rating.VERY_GOOD)
lexicon.append('extraordinary',None,Rating.VERY_GOOD)
lexicon.append('outshining',None,Rating.VERY_GOOD)
lexicon.append('fine',None,Rating.GOOD)
lexicon.append('impressed',None,Rating.GOOD)
lexicon.append('best',None,Rating.VERY_GOOD)
lexicon.append('nice',None,Rating.SOMEWHAT_GOOD)
lexicon.append('nicest',None,Rating.GOOD)
lexicon.append('complaints',None,Rating.BAD)
lexicon.append('gross',None,Rating.BAD)
lexicon.append('grossed',None,Rating.BAD)
lexicon.append('mistake',None,Rating.BAD)
lexicon.append('vain',None,Rating.BAD)
lexicon.append('sick',None,Rating.BAD)
lexicon.append('never',None,Rating.BAD)
lexicon.append('station',Topic.AMBIENCE,None)
lexicon.append('meals',Topic.FOOD,None)
lexicon.append('stuff',Topic.FOOD,None)
lexicon.append('menu',Topic.FOOD,None)
lexicon.append('delicious',Topic.FOOD,Rating.GOOD)
lexicon.append('melted',Topic.FOOD,Rating.BAD)
lexicon.append('restaurant',Topic.AMBIENCE,None)
lexicon.append('reminds',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('modern',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('hip',Topic.AMBIENCE,Rating.GOOD)
lexicon.append('atmosphere',Topic.AMBIENCE,None)
lexicon.append('bars',Topic.AMBIENCE,None)
lexicon.append('discount',Topic.VALUE,Rating.GOOD)



INTENSIFIERS = {
    'really',
    'terribly',
    'very',
    'too',
    'surely',
    'totally',
    'absolutely',
    'definitely',
    'certainly',
    'always',
    'consistently',
    'absolutely',
    'totally',
    'highly',
}

def is_intensifier(token: Token) -> bool:
    return token.lemma_.lower() in INTENSIFIERS

DIMINISHERS = {
    'barely',
    'slightly',
    'somewhat',
    'little',
    'kinda',
    'maybe',
    'mostly',
    'sorta',
    'sort of',    
}

def is_diminisher(token: Token) -> bool:
    return token.lemma_.lower() in DIMINISHERS

def signum(value) -> int:
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

_MIN_RATING_VALUE = Rating.VERY_BAD.value
_MAX_RATING_VALUE = Rating.VERY_GOOD.value


def _ranged_rating(rating_value: int) -> Rating:
    return Rating(min(_MAX_RATING_VALUE, max(_MIN_RATING_VALUE, rating_value)))

def diminished(rating: Rating) -> Rating:
    if abs(rating.value) > 1:
        return _ranged_rating(rating.value - signum(rating.value))
    else:
        return rating

def intensified(rating: Rating) -> Rating:
    if abs(rating.value) > 1:
        return _ranged_rating(rating.value + signum(rating.value))
    else:
        return rating
    
NEGATIONS = {
    'no',
    'not',
    'none',
    'lack of',
    'did not',
    'didnt',
    'dont',
    'do not',
    'cannot',
    'would not',
    'far from',
    'nothing',
}

def is_negation(token: Token) -> bool:
    return token.lemma_.lower() in NEGATIONS


_RATING_TO_NEGATED_RATING_MAP = {
    Rating.VERY_BAD     : Rating.SOMEWHAT_GOOD,
    Rating.BAD          : Rating.GOOD,
    Rating.SOMEWHAT_BAD : Rating.GOOD,  # hypothetical?
    Rating.SOMEWHAT_GOOD: Rating.BAD,  # hypothetical?
    Rating.GOOD         : Rating.BAD,
    Rating.VERY_GOOD    : Rating.SOMEWHAT_BAD,
}

def negated_rating(rating: Rating) -> Rating:
    assert rating is not None
    return _RATING_TO_NEGATED_RATING_MAP[rating]
#
#Token.set_extension('topic', default=None)
#Token.set_extension('rating', default=None)
#Token.set_extension('is_negation', default=False)
#Token.set_extension('is_intensifier', default=False)
#Token.set_extension('is_diminisher', default=False)


def debugged_token(token: Token) -> str:
    result = 'Token(%s, lemma=%s' % (token.text, token.lemma_)
    if token._.topic is not None:
        result += ', topic=' + token._.topic.name
    if token._.rating is not None:
        result += ', rating=' + token._.rating.name
    if token._.is_diminisher:
        result += ', diminisher'
    if token._.is_intensifier:
        result += ', intensifier'
    if token._.is_negation:
        result += ', negation'
    result += ')'
    return result



def opinion_matcher(doc):
    for sentence in doc.sents:
        for token in sentence:
            if is_intensifier(token):
                token._.is_intensifier = True
            elif is_diminisher(token):
                token._.is_diminisher = True
            elif is_negation(token):
                token._.is_negation = True
            else:
                lexicon_entry = lexicon.lexicon_entry_for(token)
                if lexicon_entry is not None:
                    token._.rating = lexicon_entry.rating
                    token._.topic = lexicon_entry.topic
    return doc

if nlp_en.has_pipe('opinion_matcher'):
    nlp_en.remove_pipe('opinion_matcher')
nlp_en.add_pipe(opinion_matcher)


def is_essential(token: Token) -> bool:
    return token._.topic is not None \
        or token._.rating is not None \
        or token._.is_diminisher \
        or token._.is_intensifier \
        or token._.is_negation
        
def essential_tokens(tokens):
    return [token for token in tokens if is_essential(token)]



def is_rating_modifier(token: Token):
    return token._.is_diminisher \
        or token._.is_intensifier \
        or token._.is_negation
        
def combine_ratings(tokens):
    # Find the first rating (if any).
    rating_token_index = next(
        (
            token_index for token_index in range(len(tokens))
            if tokens[token_index]._.rating is not None
        ),
        None  # Default if no rating token can be found
        
    )

    if rating_token_index is not None:
        # Apply modifiers to the left on the rating.
        original_rating_token = tokens[rating_token_index]
        combined_rating = original_rating_token._.rating
        modifier_token_index = rating_token_index - 1
        modified = True  # Did the last iteration modify anything?
        while modified and modifier_token_index >= 0:
            modifier_token = tokens[modifier_token_index]
            if is_intensifier(modifier_token):
                combined_rating = intensified(combined_rating)
            elif is_diminisher(modifier_token):
                combined_rating = diminished(combined_rating)
            elif is_negation(modifier_token):
                combined_rating = negated_rating(combined_rating)
            else:
                # We are done, no more modifiers 
                # to the left of this rating.
                modified = False
            if modified:
                # Discord the current modifier 
                # and move on to the token on the left.
                del tokens[modifier_token_index]
                modifier_token_index -= 1
        original_rating_token._.rating = combined_rating


from typing import List, Tuple  # for fancy type hints

def topic_and_rating_of(tokens: List[Token]) -> Tuple[Topic, Rating]:
    result_topic = None
    result_rating = None
    opinion_essence = essential_tokens(tokens)
    # print('  1: ', opinion_essence)
    combine_ratings(opinion_essence)
    # print('  2: ', opinion_essence)
    for token in opinion_essence:
        # print(debugged_token(token))
        if (token._.topic is not None) and (result_topic is None):
            result_topic = token._.topic
        if (token._.rating is not None) and (result_rating is None):
            result_rating = token._.rating
        if (result_topic is not None) and (result_rating is not None):
            break
    return result_topic, result_rating

def opinions(feedback_text: str):
    feedback = nlp_en(feedback_text)
    for tokens in feedback.sents:
        yield(topic_and_rating_of(tokens))
        

feedback_text = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
review = feedback_text.iloc[:, 0].values

with open('Aspect_Sentiment.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for topic, rating in opinions(str(review)):
        tsv_writer.writerow([str(topic), str(rating)])
    out_file.close()
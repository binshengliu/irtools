# import readability
import pandas as pd
import textstat
import sys


def readability(queries):
    scores = pd.DataFrame(columns=[
        'Flesch', 'Smog', 'Flesch grade', 'Coleman', 'Automated', 'Dale',
        'Difficult', 'Linsear', 'Gunning', 'Text Standard'
    ])

    scores = {
        'Flesch': [],
        'Smog': [],
        'Flesch grade': [],
        'Coleman': [],
        'Automated': [],
        'Dale': [],
        'Difficult': [],
        'Linsear': [],
        'Gunning': [],
        'Text Standard': []
    }
    for line in queries:
        # results = readability.getmeasures(line, lang='en')
        # frescores.append(results['readability grades']['FleschReadingEase'])
        # line = 'yao family wines . yao family wines is a napa valley producer founded in 2011 by yao ming , the chinese-born , five-time nba all star . now retired from the houston rockets , yao ming is the majority owner in yao family wines , which has entered the wine market with a luxury cabernet sauvignon sourced from napa valley vineyards .'
        scores['Flesch'].append(textstat.flesch_reading_ease(line))
        scores['Smog'].append(textstat.smog_index(line))
        scores['Flesch grade'].append(textstat.flesch_kincaid_grade(line))
        scores['Coleman'].append(textstat.coleman_liau_index(line))
        scores['Automated'].append(textstat.automated_readability_index(line))
        scores['Dale'].append(textstat.dale_chall_readability_score(line))
        scores['Difficult'].append(textstat.difficult_words(line))
        scores['Linsear'].append(textstat.linsear_write_formula(line))
        scores['Gunning'].append(textstat.gunning_fog(line))
        scores['Text Standard'].append(
            textstat.text_standard(line, float_output=True))

    return scores


def main():
    scores = readability(sys.stdin)
    print(scores.mean())

from pyumls import api
import pandas as pd
import csv


def scopus():
    df = pd.read_csv('scopus.csv')
    words_in_articles = df[['ID']]
    titles = df[['TI']]
    for index, words in words_in_articles.iterrows():
        # each document
        word_list = words[0].split(';')
        write_in = {}
        # remove duplicate
        word_list = list(set(word_list))
        keywords_w_cui = ''
        keywords_wo_cui = ''
        for word in word_list:
            search_result = api.search(word, apikey='5eecb034-bf01-48d3-9a8c-a893723e8bf7')
            print(word+'****')
            if len(search_result) > 0 :
                result = search_result[0]
                # print(result['ui'], result['name'])
                detail = api.getByCUI(result['ui'], apikey='5eecb034-bf01-48d3-9a8c-a893723e8bf7')
                semantic_type = detail['semanticTypes'][0]['name']
                # print(a)
                keywords_w_cui += word+'<'+result['name']+'|'+result['ui']+'|'+semantic_type+'>;'
            else:
                keywords_wo_cui += word+';'

        print(titles.iloc[index].values)

        write_in['fileName'] = titles.iloc[index].values
        write_in['keywords_w_cui'] = keywords_w_cui
        write_in['keywords_wo_cui'] = keywords_wo_cui
        writer.writerow(write_in)
        break # comment out for all document


if __name__ == '__main__':
    with open('scopus_result.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['fileName', 'keywords_w_cui', 'keywords_wo_cui'])
        writer.writeheader()
        scopus()
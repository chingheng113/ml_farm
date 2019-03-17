import xml.etree.cElementTree as ET
import glob
import errno
import csv
import os


def cas_to_csv(root, writer, file_name):
    result = {}
    targets = ['ProcedureMention', 'SignSymptomMention', 'MedicationMention', 'DiseaseDisorderMention']
    for target in targets:
        dic = {}
        print(target+'--')
        for elem in tree.iter(tag='org.apache.ctakes.typesystem.type.textsem.'+target):
            ref_ontologyConceptArr = elem.attrib['_ref_ontologyConceptArr']
            begin = elem.attrib['begin']
            end = elem.attrib['end']
            for i in root.find("uima.cas.FSArray[@_id='"+ref_ontologyConceptArr+"']").iter("i"):
                umlsConcept_id = i.text
                break
            cui = root.find("org.apache.ctakes.typesystem.type.refsem.UmlsConcept[@_id='"+umlsConcept_id+"']").get("cui")
            text = root.findall("org.apache.ctakes.typesystem.type.syntax.ConllDependencyNode[@begin='"+begin+"']")[-1].get("lemma")
            dic[cui] = text
        result['fileName'] = file_name
        result[target] = dict_to_line(dic)
    writer.writerow(result)
    return dic


def dict_to_line(dict):
    line=''
    for key, value in dict.items():
        line += value+'<'+key+'>'+';'
    return line




if __name__ == '__main__':
    mypath = '/Users/aalinc9/Desktop/test/*.xml'
    files = glob.glob(mypath)
    with open('cas_result.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['fileName', 'ProcedureMention', 'SignSymptomMention',
                                                      'MedicationMention', 'DiseaseDisorderMention'])
        writer.writeheader()
        for file in files:
            try:
                file_name = os.path.basename(file)
                tree = ET.ElementTree(file=file)
                root = tree.getroot()
                cas_to_csv(root, writer, file_name)
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
            print("###")
    print('done')

import logging
import os
from tqdm import tqdm
import PyPDF2
import fitz
import pandas as pd


def get_file_list(dir, file_type_list=['pdf'], file_list=[]):
    for root, _, files in os.walk(dir):
        for file in files:
            file_type = file[file.rfind('.') + 1:]
            if file_type in file_type_list:
                file_list.append(os.path.join(root, file))
    return file_list


if __name__ == '__main__':
    pdf_path = 'data/pdf/' # save all pdf file with a fixed name
    # file name like: "ID_Year.pdf", such as 1_2019.pdf
    file_paths = get_file_list(pdf_path, ['pdf'])  # 获取文件列表
    data = open('data/data.txt', 'w+', encoding='utf-8')
    datalist = []
    for file in tqdm(file_paths, ascii=True, desc="PDF To TXT: "):
        try:
            filename = str(file).split('\\')[1].split('.')[0]
            ID = int(filename.split('_')[0])

            pdf_file = fitz.open(file)
            d = []
            for page in pdf_file:
                text = page.get_text("text")
                findtext = str(text).replace('\n', ' ')
                findtext = str(findtext).replace('\r', ' ')
                findtext = str(findtext).replace(' ', ' ')
                findtext = str(findtext).replace(' ', ' ')
                findtext = str(findtext).replace(' ', ' ')
                findtext = str(findtext).replace(' ', ' ')
                findtext = str(findtext).replace('­', ' ')
                findtext = str(findtext).replace('           ', '')
                findtext = str(findtext).replace('         ', '')
                findtext = str(findtext).replace('         ', '')
                findtext = str(findtext).replace('       ', '')
                findtext = str(findtext).replace('  ', '')
                findtext = str(findtext).replace('	', '')
                d.append(findtext)
            data.write(' '.join(d) + '\n')
            datalist.append([ID, filename])
        except:
            print('Error, %s' % file)

    pd.DataFrame(datalist).to_csv('data/right_transform.csv', header=['ID', 'Name'], index=False)
    data.close()

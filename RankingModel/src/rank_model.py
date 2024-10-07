# Задание:
# Для каждой услуги из нового прайса клиники: service_name (main feature)
# Предложить топ-5 наиболее подходящих эталонных услуг: local_name (target)

# Данные из эталонного прайса
# local_data = {
#     "local_id": 5927.0,
#     "type": "laboratory_tests",  # Категория услуги
#     "gt_type_name": "Биохимия",  # Подкатегория услуги
#     "parent_id_name": "Лабораторная диагностика",  # Родительская категория
#     "local_name": "Varicella Zoster Virus, IgG"  # Эталонное название услуги
# }

# # Данные клиники
# clinic_data = {
#     "lpu_name": "Инвитро",  # Название клиники
#     "service_name": "Антитела класса IgG к вирусу Varicella-Zoster"  # Название услуги в клинике
# }

# Расшифровка признаков: 
# local_id - Уникальный идентификатор услуги в эталонном прайсе;
# type - Категория или тип услуги, например, лабораторные тесты;
# gt_type_name - Подкатегория услуги, например, Биохимия;
# parent_id_name - Родительская категория услуги, например, Лабораторная диагностика;
# local_name - Эталонное название услуги;
# lpu_name - Название клиники;
# service_name - Название услуги в клинике;





import numpy as np
from tqdm import tqdm
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from sklearn.metrics.pairwise import cosine_similarity




class RankSystem():
    def __init__(self, 
                 path_to_file = './merged_df.csv',
                 loaded_llm = True,
                 ) -> None:
        
        self.path_to_file = path_to_file
        self.data = pd.read_csv(self.path_to_file, sep=',', encoding='utf-8')
        self.loaded_llm = loaded_llm
        print(f'Исходный датасет:\n{self.data.head(3)}\n')

        if self.loaded_llm == False:
            self.bge_m3 = HuggingFaceEmbeddings(model_name = './models/embedding/')

        # прочие атрибуты класса
        self.embedding_data_dict = None
        self.service_features = None
        self.target_features = None
        self.score_matrix = None


    def eda(self,):
        
        print(f'общая информация по dataset')
        self.data.info()
        print('\n')
        # проверим какие столбцы содержат пропуски
        print(f'столбцы со значением None:\n{self.data.isna().any(axis=0)}\n')

        # строки с пропусками
        print(f'строки со значением None:\n{self.data[self.data.isna().any(axis=1)]}\n')

        # переведем local_id эталонных услуг к int
        self.data['local_id'] = self.data['local_id'].astype(int)

        print("""В задаче важную роль играют две колонки: local_name и service_name.
Другие признаки в целом отражают дополнительную информацию и могут быть использованы в описании рекомендованных услуг.
Пропущеннные значения в local_name и service_name отсутствуют\n""")
        
        # отберем необходимые столбцы
        columns_names = list(self.data.keys())
        print(f'columns_names:\n{columns_names}\n')

        # local_id относится к local_name в эталонном прайсе
        clean_data = self.data[['local_id', 'local_name', 'service_name', 'lpu_name']].copy()
        clean_data.isna().any()

        # число уникальных услуг
        local_name = clean_data['local_name'].unique()
        service_name = clean_data['service_name'].unique()
        print(
            f"""Число уникальных local_name услуг:{clean_data['local_name'].unique().__len__()}, 
Число уникальных service_name услуги: {clean_data['service_name'].unique().__len__()}"""
        )

        # найдем эмбеддинги local_name, service_name  
        if self.loaded_llm == False:
            embedding_dict = {
            'local_name': self.build_embedding(local_name),
            'service_name': self.build_embedding(service_name)
            }

            with open('./embedding_dict.pickle', 'wb') as file:
                pickle.dump(embedding_dict, file)
            with open('./embedding_dict.pickle', 'rb') as file:
                self.embedding_data_dict = pickle.load(file)
        else:
            with open('./embedding_dict.pickle', 'rb') as file:
                self.embedding_data_dict = pickle.load(file)
        # print(f'embedding_data_dict: \n{self.embedding_data_dict}')

    def build_embedding(self, data):
        embeddings = {}
        for service in tqdm(data, mininterval = 0.5, desc='Total', total=data.__len__()):
            # нормализация текста
            service = service.lower().strip()
            # построение вектора
            embeddings[service] = self.bge_m3.embed_query(service)
        return embeddings
    
    def build_dataset(self,):

        # feature_i -> [target_0, ..., target_4]
        # разделяем услуги клиники от эталонных услуг из прейскуранта
        # каждая услуга есть пара (описание услуги, embedding)
        self.service_features = [(k,v) for k,v in self.embedding_data_dict['service_name'].items()]
        self.target_features = [(k,v) for k,v in self.embedding_data_dict['local_name'].items()]
        del self.embedding_data_dict

        # формирование матрицы признаков и таргетов
        features_array = np.array([service[1] for service in self.service_features])
        target_array = np.array([service[1] for service in self.target_features])

        # score_matrix[i,j] -> similarity score для i-услуги клиники и j-эталонной услуги из прайса
        self.score_matrix = cosine_similarity(features_array, target_array)
        print(f'размерность матрицы схожести: {self.score_matrix.shape}\n')
        return self.service_features, self.target_features

    def make_prediction(self, 
                        TOP_K,
                        service_name_idx,
                        ):
        service_title = self.service_features[service_name_idx][0]
        # print(f'название услуги: {service_title}\n')

        ranked_services_list = sorted(
            [(idx, score) for idx, score in enumerate(self.score_matrix[service_name_idx])],
            reverse=True,
            key=lambda x: x[1]
        )[:TOP_K]

        ranked_services = []
        for itm in ranked_services_list:
            ranked_services.append((self.target_features[itm[0]][0], f'score: {round(itm[1],3)}'))

        print(f'Услуга из прайса клиники: {service_title}\nCписок эталонных услуг: {ranked_services}')
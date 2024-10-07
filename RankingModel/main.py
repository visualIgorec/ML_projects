import pickle
from src.rank_model import RankSystem


if __name__ == "__main__":
    model = RankSystem(
        path_to_file = './merged_df.csv',
        loaded_llm = True
    )
    model.eda()
    service_features, target_features = model.build_dataset()
    with open('./service_features.pickle', 'wb') as file:
        pickle.dump(service_features, file)
    with open('./target_features.pickle', 'wb') as file:
        pickle.dump(target_features, file)

    # покажем название услуги в клинике
    TOP_K = 5
    desc_service = [i[0] for i in service_features]
    example = [
        'd-димер',
        'комплексное исследование «стресс»',
        'холестерин общий (холестерин, cholesterol total)'
    ]
    for idx, context in enumerate(example):
        print(f'\nprediction #{idx}')
        service_name_idx = desc_service.index(context)
        model.make_prediction(
            TOP_K,
            service_name_idx
        )
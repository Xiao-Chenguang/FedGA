def agg_model(models):
    for m in models[1:]:
        models[0] + m
    models[0] / len(models)
    return models[0]

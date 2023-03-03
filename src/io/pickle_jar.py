import pickle


def pickle_object(pkl_path, obj_to_pkl):
    with open(pkl_path, "wb") as pkl:
        pickle.dump(obj_to_pkl, pkl, pickle.HIGHEST_PROTOCOL)
    return None


def unpickle_object(pkl_path):
    with open(pkl_path, "rb") as pkl:
        unpickled_obj = pickle.load(pkl)
    return unpickled_obj

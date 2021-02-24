import argparse
import pickle
import time

import dill


def convert(old_pkl, root="img_features/python2_pickles/"):

    old_pkl_path = root + old_pkl

    dill._dill._reverse_typemap["ObjectType"] = object
    # Open the pickle using latin1 encoding
    with open(old_pkl_path, "rb") as f:
        loaded = pickle.load(f, encoding="bytes")

    new_data = []
    for item in loaded:
        new_item = {}
        import pdb

        pdb.set_trace()
        for key, value in item.items():
            new_key = key.decode() if isinstance(key, bytes) else key
            new_value = value.decode() if isinstance(value, bytes) else value

            if isinstance(new_value, str) and new_key not in ["scanId", "viewpointId"]:
                new_value = eval(new_value)

            new_item[new_key] = new_value

            print(type(new_key), new_key)
            print(type(new_value), new_value)

        new_data.append(new_item)
    new_pkl = f"img_features/{old_pkl}"
    with open(new_pkl, "wb") as outfile:
        pickle.dump(new_data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Python 2 pickle filename")
    args = parser.parse_args()
    start = time.time()
    convert(args.infile)
    now = time.time()
    print(
        "Time taken for converting the pickle file: %0.4f mins" % ((now - start) / 60)
    )

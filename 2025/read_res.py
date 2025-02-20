import gzip, pickle
import numpy as np

def read_file(file_name):
    with gzip.open(file_name, "rb") as f:
        ARRIVED, COLLISION, TIMEOUT, ARRIVED_TIMES, COLLISION_TIMES = pickle.load(f)
    TRY_TIMES = ARRIVED + COLLISION + TIMEOUT
    print(f"Arrived: {ARRIVED}/{TRY_TIMES}, Collision: {COLLISION}/{TRY_TIMES}, Overtime: {TIMEOUT}/{TRY_TIMES}")
    print(f"Average Arrived times: {np.sum(ARRIVED_TIMES)/ARRIVED}")
    print(f"Average Collision times: {np.sum(COLLISION_TIMES)/COLLISION}")
    return (ARRIVED, COLLISION, TIMEOUT, np.sum(ARRIVED_TIMES)/ARRIVED, np.sum(COLLISION_TIMES)/COLLISION)

if __name__ == "__main__":
    import os, sys
    path = sys.argv[1]

    res_files = []
    # get all files name list in the path
    files = os.listdir(path)
    for file in files:
        if file.endswith("res2025.gzip"):
            res_files.append(file)
    
    res = []
    for file in res_files:
        print(f"File: {file}")
        single_res = read_file(os.path.join(path, file))
        res.append((file, single_res))
    
    def compare(item1, item2):
        r1 = item1[1]
        r2 = item2[1]
        # arrived times
        if r1[0] != r2[0]:
            return r1[0] - r2[0]
        # collision times
        if r1[1] != r2[1]:
            return r2[1] - r1[1]
        # timeout times
        if r1[2] != r2[2]:
            return r2[2] - r1[2]
        # average arrived times
        if r1[3] != r2[3]:
            return r2[3] - r1[3]
        # average collision times
        if r1[4] != r2[4]:
            return r1[4] - r2[4]
        return 0
    
    from functools import cmp_to_key
    sorted_res = sorted(res, key=cmp_to_key(compare))

    print("Sorted result:\n--------------------------------")
    for item in sorted_res:
        output = []
        output.append(item[0])
        output.extend(item[1])
        print('\t'.join([str(i) for i in output]))
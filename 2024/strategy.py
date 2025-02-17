import random
CONFIG = {
    "seed" : random.randrange(1000),
    "speed_up" : 1,
}
# judgement condition : 
# - have_targets
# - x_larger_than, Value
# - x_smaller_than, Value
# - y_larger_than, Value
# - y_smaller_than, Value

# action:
# - to_point, x, y
# - to_closest_target

# ROOT - have_targets - to_closest_target
#      - x_larger_than, 0.0 - y_larger_than, 0.0 - to_point, IMPORTANT_D, -0.01
#                           - to_point, -0.01, -IMPORTANT_D
#      - y_larger_than, 0.0 - to_point, 0.01, IMPORTANT_D
#      - to_point, -IMPORTANT_D, 0.01
DECISION_TREE = [
    {
        "ID" : "HAVE_TARGET",
        "Type" : "Judgement",
        "Arguments" : ["have_targets"],
        "Children" : [
            {
                "ID" : "TO_CLOSEST",
                "Type" : "Action",
                "Arguments" : ["to_closest_target"],
            }
        ],
    },
]
CONFIG = {
    "ID" : "markmark",
    "seed" : 0,
    "speed_up" : 10,
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

IMPORTANT_D = 0.71

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
    {
        "ID" : "X_LARGER_THAN",
        "Type" : "Judgement",
        "Arguments" : ["x_larger_than", 0.0],
        "Children" : [
            {
                "ID" : "Y_LARGER_THAN",
                "Type" : "Judgement",
                "Arguments" : ["y_larger_than", 0.0],
                "Children" : [
                    {
                        "ID" : "TO_POINT_1",
                        "Type" : "Action",
                        "Arguments" : ["to_point", IMPORTANT_D, -0.01],
                    },
                ],
            },{
                "ID" : "TO_POINT_2",
                "Type" : "Action",
                "Arguments" : ["to_point", -0.01, -IMPORTANT_D],
            }
        ],
    },{
        "ID" : "Y_LARGER_THAN",
        "Type" : "Judgement",
        "Arguments" : ["y_larger_than", 0.0],
        "Children" : [
            {
                "ID" : "TO_POINT_3",
                "Type" : "Action",
                "Arguments" : ["to_point", 0.01, IMPORTANT_D],
            },
        ],
    },{
        "ID" : "TO_POINT_4",
        "Type" : "Action",
        "Arguments" : ["to_point", -IMPORTANT_D, 0.01],
    },
]
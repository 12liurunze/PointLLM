mc_sim_7b_63 = [[0],[1],[2],[3],[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]
                ,[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
                [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,0,0],[0,0,0,0,1]]

simple_depth5 = [[0], [1], [0, 0], [0, 1], [1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
mc_sim_7b_9_depth5 = simple_depth5

# Wider PointLLM trees. They raise MAT by covering more plausible shallow
# branches while retaining the high-probability deep paths from mc_sim_7b_63.
pointllm_wide_35 = mc_sim_7b_63 + [
    [4], [5],
    [0, 3], [1, 2], [2, 2], [3, 1], [4, 0],
    [1, 0, 1], [1, 1, 0], [2, 0, 0],
]

pointllm_wide_45 = pointllm_wide_35 + [
    [6], [7],
    [0, 4], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0],
    [0, 0, 3], [0, 1, 2],
]

# Wider verification trees for full-point PointLLM. The added nodes extend
# shallow rank coverage before spending nodes on deeper, lower-probability paths.
pointllm_wide_55 = pointllm_wide_45 + [
    [8], [9],
    [0, 5], [1, 4], [2, 4], [3, 3], [4, 2], [5, 1], [6, 0],
    [0, 1, 3],
]

pointllm_wide_65 = pointllm_wide_55 + [
    [0, 0, 4], [0, 2, 2], [1, 0, 2], [1, 1, 1], [2, 0, 1],
    [3, 0, 1], [4, 0, 0], [5, 0, 0],
    [0, 0, 0, 3], [0, 0, 1, 0],
]

# Path-budgeted tree learned from PointLLM acceptance traces. It spends the
# extra verification nodes on descendants of frequently accepted paths.
pointllm_path_55 = pointllm_wide_45 + [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 2, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 2, 0, 0],
]

pointllm_path_65 = pointllm_path_55 + [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [2, 0, 0, 0],
    [3, 0, 0],
]

# Prefix-closed tree learned from accepted PointLLM draft paths.
pointllm_learned_35 = [
    [0], [0, 0], [1], [0, 0, 0], [0, 1], [2], [1, 0], [0, 2], [3],
    [0, 0, 1], [0, 1, 0], [0, 3], [0, 0, 0, 0], [5], [0, 4],
    [0, 2, 0], [2, 0], [4], [1, 0, 0], [3, 0], [1, 1], [7], [8],
    [5, 0], [0, 0, 0, 0, 0], [2, 0, 0], [6], [9], [1, 2],
    [1, 0, 1], [1, 3], [2, 1], [4, 0], [0, 1, 1], [6, 0],
]

import numpy as np
from utilities import timer

def mutate_weights(weights: list[dict]):
    
    def crossover(icouple: list[int, int]):
        new_weights = {}
        for layer in layers_name:
            new_weights[layer] = {
                'weight': np.where(
                        np.random.randint(0,2,parents[icouple[0]][layer]['weight'].shape).astype(bool),
                        parents[icouple[0]][layer]['weight'],
                        parents[icouple[1]][layer]['weight'],
                    ),
                'bias': np.where(
                        np.random.randint(0,2,parents[icouple[0]][layer]['bias'].shape).astype(bool),
                        parents[icouple[0]][layer]['bias'],
                        parents[icouple[1]][layer]['bias'],
                    ),
                'activation': parents[icouple[0]][layer]['activation'],
            }
        return new_weights
    
    def mutation(child):
        MUTATION_INTENSITY = 0.1
        new_weights = {}
        for layer in layers_name:
            new_weights[layer] = {    
                'weight': child[layer]['weight'] + np.random.uniform(
                        -MUTATION_INTENSITY, MUTATION_INTENSITY,
                        child[layer]['weight'].shape,
                    ),
                'bias': child[layer]['bias'] + np.random.uniform(
                        -MUTATION_INTENSITY, MUTATION_INTENSITY,
                        child[layer]['bias'].shape,
                    ),
                'activation': child[layer]['activation'],
            }
        return new_weights
    
    parents = weights[:weights.shape[0]//2]
    parent_pool = np.arange(parents.shape[0])
    np.random.shuffle(parent_pool)
    parent_pool = parent_pool.reshape(-1,2)
    layers_name = [layer for layer in parents[0].keys()]
    childs = list(map(crossover, parent_pool))
    mutated_childs = list(map(mutation, childs))
    return np.concatenate([parents, childs, mutated_childs])
    
    
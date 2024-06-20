import json
import tqdm
from pathlib import Path


from kaggle_arc.arc_interface import Board, BoardPair, Riddle

def read_single_dataset(basedir, prefix) -> list[Riddle]:
    if isinstance(basedir, str):
        basedir = Path(basedir)
    challenges_file = basedir / f"{prefix}_challenges.json"
    solutions_file = basedir / f"{prefix}_solutions.json"
    if not challenges_file.exists():
        return None
    with open(challenges_file.as_posix()) as f:
        riddles = json.load(f)
    solutions = {}
    if solutions_file.exists():
        with open(solutions_file.as_posix()) as f:
            solutions = json.load(f)

    riddles_list = []
    for key, task in tqdm.tqdm(riddles.items()):
    
        test = task['test']
        train = task['train']

        train_pairs = []

        for i in train:
            ib = Board(root=i['input'])
            ob = Board(root=i['output'])
            train_pairs.append(BoardPair(input=ib, output=ob))

        test_pairs = []

        for i in test:

            ib = Board(root=i['input'])
            ob = Board(root=solutions[key][0])
            test_pairs.append(BoardPair(input=ib, output=ob))
        
        r = Riddle(test=test_pairs, train=train_pairs, riddle_id=key)
        riddles_list.append(r)
    return riddles_list


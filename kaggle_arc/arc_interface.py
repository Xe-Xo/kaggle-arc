import pydantic
from typing import Optional

import itertools
import json
import numpy as np
from colored import attr, bg, fg
from matplotlib import pyplot as plt


from arc_types import Grid, GridList, List, Tuple

COLORMAP = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3, 5: 8, 6: 5, 7: 166, 8: 6, 9: 52}
CELL_PADDING_STR = " " * 1
BOARD_GAP_STR = " " * 5
PAIR_GAP_STR = "\n" + " " * 1 + "\n"


def listit(t):
    # Iterates through a list of lists and converts them to tuples.
    return tuple(map(listit, t)) if isinstance(t, (List, Tuple)) else t

class Board(pydantic.BaseModel):

    """
    A list of a list of integers that represent either the input or the output state.
    """
    __root__: GridList

    @property
    def data(self) -> Grid:
        return listit(self.__root__)
    
    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_cols(self) -> int:
        return len(self.data[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    @property
    def flat(self) -> list[int]:
        return list(itertools.chain.from_iterable(self.data))

    @property
    def unique_values(self) -> set[int]:
        return set(self.flat)

    @property
    def num_unique_values(self) -> int:
        return len(self.unique_values)
    
    @property
    def np(self) -> np.ndarray:
        return np.array(self.data, dtype=np.int64)    

    def fmt_cell(self, row: int, col: int, colored=False) -> str:
        value = self.data[row][col]
        color = COLORMAP[value]
        value_str = f"{CELL_PADDING_STR}{value}{CELL_PADDING_STR}"
        if colored:
            return f"{fg(15)}{bg(color)}{value_str}{attr(0)}"
        else:
            return value_str

    def fmt_row(self, row: int, colored=False) -> str:
        return "".join(
            self.fmt_cell(row, col, colored=colored)
            for col in range(len(self.data[row]))
        )

    def fmt_empty_row(self):
        return "".join(
            f"{CELL_PADDING_STR} {CELL_PADDING_STR}" for _ in range(self.num_cols)
        )

    def fmt(self, colored=False) -> str:
        return "\n".join(
            self.fmt_row(row, colored=colored) for row in range(len(self.data))
        )

class BoardPair(pydantic.BaseModel):
    input: Board
    output: Board

    @property
    def dictionary(self):
        return {
            'input': self.input.__root__,
            'output': self.input.__root__
        }

    def fmt(self, colored=False, with_output=True) -> str:
        rows = []
        max_row = (
            max(self.input.num_rows, self.output.num_rows)
            if with_output
            else self.input.num_rows
        )
        for row in range(max_row):
            row_parts = []
            if row >= self.input.num_rows:
                row_parts.append(self.input.fmt_empty_row())
            else:
                row_parts.append(self.input.fmt_row(row, colored=colored))
            if with_output:
                row_parts.append(BOARD_GAP_STR)
                if row >= self.output.num_rows:
                    row_parts.append(self.output.fmt_empty_row())
                else:
                    row_parts.append(self.output.fmt_row(row, colored=colored))
            rows.append("".join(row_parts))
        return "\n".join(rows)

    def as_np(self, with_solution=True):
        return (self.input.np, self.output.np if with_solution else None)

class Riddle(pydantic.BaseModel):
    train: list[BoardPair]
    test: list[BoardPair]
    riddle_id: Optional[str] = None



    @property
    def dictionary(self):
        return {
            'train': [x.dictionary for x in self.train],
            'test': [x.dictionary for x in self.test]
        }


    def save_json(self,subdir):

        with open(f'data/{subdir}/{self.riddle_id}.json', 'w', encoding='utf-8') as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=4)
            

    def fmt_plt(self, with_test_outputs=False):
        parts = [pair.as_np() for pair in self.train]
        parts.extend(pair.as_np(with_solution=with_test_outputs) for pair in self.test)

        def _draw_board(board: Optional[np.ndarray], ax: plt.Axes):
            if board is None:
                ax.remove()
            else:
                ax.matshow(board, cmap="nipy_spectral", vmin=0.0, vmax=9.0)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        fig, axs = plt.subplots(len(parts), 2)
        for (inboard, outboard), (inax, outax) in zip(parts, axs):
            _draw_board(inboard, inax)
            _draw_board(outboard, outax)

        plt.tight_layout()
        return fig, axs





import tkinter as tk
import numpy as np
import matplotlib as mpl
import matplotlib.cm as mplcm


class GridWorld:

    def __init__ (self, width, height):
        self.width = width
        self.height = height
        self.num_cells = width*height
        self._active = np.ones((height, width), dtype=np.bool)
        self._graph = np.zeros((height, width, height, width))

        def connect (x0, y0, x1, y1, w):
            if (0 <= x0 < width and 0 <= y0 < height and
                0 <= x1 < width and 0 <= y1 < height):
                self._graph[y0, x0, y1, x1] = w
                self._graph[y1, x1, y0, x0] = w

        for y in range(height):
            for x in range(width):
                for (x1, y1) in ((x+1, y), (x, y+1)):
                    connect(x, y, x1, y1, 0.25)

    def __getitem__ (self, coord):
        (x, y) = coord
        return self._active[y, x]

    def __setitem__ (self, coord, active):
        (x, y) = coord
        self._active[y, x] = active

    def set_all (self, active):
        self._active.fill(active)


def normalized_laplacian (graph):
    sqrt_rowsums = np.sqrt(np.sum(graph, axis=0, keepdims=True))
    graph /= sqrt_rowsums
    graph /= sqrt_rowsums.T
    return np.eye(*graph.shape) - graph


class GridWorldPVF:

    def __init__ (self, gridworld):

        num_cells = gridworld.num_cells
        graph = np.reshape(gridworld._graph, (num_cells, num_cells))
        active = np.reshape(gridworld._active, (num_cells,))
        subgraph = graph[active][:, active]
        subgraph.flat[::subgraph.shape[0]+1] += 1 - np.sum(subgraph, axis=0)

        (self._eigvals, eigvecs) = np.linalg.eigh(normalized_laplacian(subgraph))
        self._eigvals[self._eigvals < 0] = 0

        eigvec_max = np.amax(np.absolute(eigvecs), axis=0, keepdims=True)
        eigvec_max[eigvec_max == 0] = 1.0
        eigvecs /= eigvec_max

        self._eigvecs = np.zeros((gridworld.height, gridworld.width, len(self)))
        np.reshape(self._eigvecs, (num_cells, len(self)))[active] = eigvecs

    def __len__ (self):
        return len(self._eigvals)

    def __getitem__ (self, i):
        if isinstance(i, slice):
            return [self[j] for j in i.indices(len(self))]
        elif 0 <= i < len(self):
            return (self._eigvals[i], self._eigvecs[..., i])
        else:
            raise IndexError("Invalid index: {}".format(i))

    def min_eigval (self):
        return np.amin(self._eigvals)

    def max_eigval (self):
        return np.amax(self._eigvals)

    def eigval_index (self, eigval):
        return np.fmin(np.searchsorted(self._eigvals, eigval), len(self))


class Application (tk.Frame):

    def __init__ (self, gridworld, cell_size, master):
        super().__init__(master)
        self._gridworld = gridworld
        self._cell_size = cell_size
        self._current_pvf = 0
        self._cmap = mplcm.ScalarMappable(norm=mpl.colors.Normalize(-1.0, 1.0),
                                          cmap='plasma')

        self._reset_button = tk.Button(self, text='Reset', command=self.reset_cells)
        self._reset_button.pack(fill=tk.X, expand=1)

        (self._canvas, self._rects) = self._make_canvas(self, cell_size,
                                                        gridworld.width,
                                                        gridworld.height)

        for ev in ("<Button-1>", "<B1-Motion>"):
            self._canvas.bind(ev, lambda event: self._handle_set_cell(event, False))

        for ev in ("<Button-3>", "<B3-Motion>"):
            self._canvas.bind(ev, lambda event: self._handle_set_cell(event, True))

        self._pvfselect = tk.Scale(self, orient=tk.HORIZONTAL, label='PVF',
                                   from_=1, command=self._handle_pvfselect)
        self._pvfselect.pack(fill=tk.X, expand=1)

        self._eigselect = tk.Scale(self, orient=tk.HORIZONTAL, label='Eigenvalue',
                                   from_=0, resolution=-1,
                                   command=self._handle_eigselect)
        self._eigselect.pack(fill=tk.X, expand=1)

        self.recalculate()
        self.pack()

    @staticmethod
    def _make_canvas (master, cell_size, width, height):
        canvas_size = np.multiply([width, height], cell_size)
        canvas = tk.Canvas(master, width=canvas_size[0], height=canvas_size[1],
                           bg='black')

        def make_rect (x, y):
            return canvas.create_rectangle(*np.multiply((x, y, x+1, y+1), cell_size))

        rects = [[make_rect(x, y) for x in range(width)] for y in range(height)]
        canvas.pack()
        return (canvas, rects)

    def _paint_cell (self, x, y, color=''):
        if not self._gridworld[x, y]:
            color = ''
        self._canvas.itemconfig(self._rects[y][x], fill=color)

    def _paint_cells (self, pvf=None):
        if pvf is not None:
            rgb = self._cmap.to_rgba(pvf)
        for y in range(self._gridworld.height):
            for x in range(self._gridworld.width):
                color = '' if pvf is None else mpl.colors.rgb2hex(rgb[y, x])
                self._paint_cell(x, y, color)

    def set_cell (self, x, y, active):
        if 0 <= x < self._gridworld.width and 0 <= y < self._gridworld.height:
            if self._gridworld[x, y] != active:
                self._gridworld[x, y] = active
                self.recalculate()

    def reset_cells (self):
        self._gridworld.set_all(True)
        self.recalculate()

    def recalculate (self):
        self._pvfs = GridWorldPVF(self._gridworld)
        self._pvfselect.config(to=len(self._pvfs))
        self._eigselect.config(to=self._pvfs.max_eigval())
        self.show_pvf(self._current_pvf, force_redraw=True)

    def show_pvf (self, index, force_redraw=False):
        (eigval, pvf) = self._pvfs[index]
        self._set_eigselect(eigval)
        if self._current_pvf != index or force_redraw:
            self._current_pvf = index
            self._paint_cells(pvf)

    def _handle_pvfselect (self, index):
        self.show_pvf(int(index) - 1)

    def _handle_eigselect (self, eigval):
        index = self._pvfs.eigval_index(float(eigval))
        eigval = self._pvfs[index][0]
        self._eigselect.set(eigval)
        self._pvfselect.set(index + 1)

    def _set_eigselect (self, eigval):
        handler = self._eigselect.cget('command')
        self._eigselect.config(command='')
        self._eigselect.set(eigval)
        self.after_idle(lambda: self._eigselect.config(command=handler))

    def _handle_set_cell (self, event, active):
        (x, y) = (event.x // self._cell_size, event.y // self._cell_size)
        self.set_cell(x, y, active)


if __name__ == '__main__':
    gridworld = GridWorld(15, 15)
    root = tk.Tk()
    root.title("Protovaluefunctionator")
    app = Application(gridworld, cell_size=40, master=root)
    app.mainloop()

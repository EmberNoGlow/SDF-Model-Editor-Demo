class History:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def add(
        self,
        undo_func,
        redo_func,
        undo_args=None,
        redo_args=None,
        undo_kwargs=None,
        redo_kwargs=None,
    ):
        if undo_args is None:
            undo_args = ()
        if redo_args is None:
            redo_args = ()
        if undo_kwargs is None:
            undo_kwargs = {}
        if redo_kwargs is None:
            redo_kwargs = {}

        self.undo_stack.append(
            (undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs)
        )
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False

        undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs = (
            self.undo_stack.pop()
        )
        undo_func(*undo_args, **undo_kwargs)
        self.redo_stack.append(
            (undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs)
        )
        return True

    def redo(self):
        if not self.redo_stack:
            return False

        undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs = (
            self.redo_stack.pop()
        )
        redo_func(*redo_args, **redo_kwargs)
        self.undo_stack.append(
            (undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs)
        )
        return True

class ReIterable:
    def __init__(self, generator_function):
        self.func = generator_function

    def __iter__(self):
        return self.func()

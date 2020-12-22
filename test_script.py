class test:
    def __init__(self) -> None:
        self.option = 1
        self.mask = True
        self.annotation = True
    
    def switcher(self):
        switch_case = {
            True : print("hello"),
            False : print("bye")
        }
        switch_case.get(self.mak)
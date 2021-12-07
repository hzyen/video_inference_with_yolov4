class finishedAlertEvent():
    def __init__(self):
        self._isFinished = False
    
    def is_set(self):
        if self._isFinished:
            return True
        return False
    
    def set(self):
        self._isFinished = True
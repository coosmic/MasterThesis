class JobInterface:

    def __init__(self):
        self.name = "JobInterface"

    def execute(self, jobParameter):
        pass
    
    def getStatus(self):
        return {}

    def getName(self):
        return self.name
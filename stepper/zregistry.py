
### Enables to save all the steppers at the same time before closing a session for instance

class StepperRegistry:
    dict_steppers={}
    def __init__(self):
        self.list_steppers=[]
    
    def add(self,stepper_class):
        self.dict_steppers[hash(stepper_class)]=stepper_class
    
    def save(self):
        for k,v in self.dict_steppers.itmes():
            v.save()
            
    def clean(self):
        for k,v in self.dict_steppers.itmes():
            v.save()
        
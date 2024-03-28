'''
The class which gathers data about MAC execution:
    - The number of memory access needed to execute a MAC
    - The number of operations needed to execute a MAC
    - The number of cycles needed to execute a MAC
'''
class MacProfile:
    def __init__(self, N_memory_access: int = 3, N_operations: int = 2, N_cycles = 2):
        self.N_memory_access = N_memory_access
        self.N_operations = N_operations
        self.N_cycles = N_cycles

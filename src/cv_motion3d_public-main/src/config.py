class Confing():
    def __init__(self) -> None:
        self.subspace_dim = 3
        self.interval = 5

        self.motion_description = {
                            '01_01':'forward jumps', 
                            '01_02':'climb',
                            '01_09':'climb',
                            '02_01':'walk',
                            '02_02':'walk',
                            '02_03':'run/jog',
                            '07_01':'walk',
                            '16_12':'walk, veer left', 
                            '16_17':'walk, 90-degree left turn', 
                            '16_57':'run/jog, sudden stop'}

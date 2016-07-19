import sys
import utils
from datetime import datetime
utils.print_generator_command_strings( 
            datetime.strptime(sys.argv[1], '%Y%m%d'), 
            datetime.strptime(sys.argv[2], '%Y%m%d'), 
            split=6, 
            verify = True,
            root_dir='/local/disk1/data2/snapshots/0/')

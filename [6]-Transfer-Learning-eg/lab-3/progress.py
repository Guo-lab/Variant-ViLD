# -*- coding: utf-8 -*-

# Print iterations progress to the terminal

import sys
def print_progress_bar(iteration, total, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    str_format = "{0:." + str(1) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    progress_bar= '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r |%s| %s%s' % (progress_bar, percents, '%')),
    # Print New Line on Complete
    if iteration == total:
        print()

#################
# Sample Usage
################
#import progress
#from progress import printProgressBar

#for i in range (456):
#	print_progress_bar(i,456)
#	performe some calculation...

################
# Sample Output
################
# |█████████████████████████████████████████████-----| 90.0%
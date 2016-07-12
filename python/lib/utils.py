# run system commands
from subprocess import call


def easy_call(command):
    try:
        call(command, shell=True)
    except Exception as exception:
        print "Error: could not execute the following"
        print ">>", command
        print type(exception)  # the exception instance
        print exception.args  # arguments stored in .args
        exit(-1)


def crop_wav(wav_path, start_trim, end_trim, output_path):
    duration = end_trim - start_trim
    cmd = 'sbin/sox %s %s trim %s %s' % (wav_path, output_path, str(start_trim), str(duration))
    easy_call(cmd)


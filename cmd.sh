# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

# more-or-let custom: feat command
export feat_cmd="slurm.pl --time 00:30:00"
export train_cmd="run.pl --gpu 2 --time 12:00:00 --mem 128000M"
export decode_cmd="run.pl --gpu 2 --time 01:00:00 --mem 128000M"
export mkgraph_cmd="run.pl"
# the use of cuda_cmd is deprecated but it's still sometimes used in nnet1
# example scripts.
export cuda_cmd="run.pl"


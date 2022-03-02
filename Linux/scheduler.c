#include <stdio.h>
#include<sched.h>
main()
{
	int scheduler;
	scheduler = sched_getscheduler(getpid());
	printf(" The scheduler policy of the current process %d\t is %d\n",getpid(),scheduler);
	printf(" The max priority value for SCHED_OTHER is %d\n",sched_get_priority_max(SCHED_OTHER));
	printf(" The min priority value for SCHED_OTHER is %d\n",sched_get_priority_min(SCHED_OTHER));
	printf(" The min priority value for SCHED_FIFO is %d\n",sched_get_priority_min(SCHED_FIFO));
	printf(" The max priority value for SCHED_FIFO is %d\n",sched_get_priority_max(SCHED_FIFO));
	printf(" The min priority value for SCHED_RR is %d\n",sched_get_priority_min(SCHED_RR));
	printf(" The max priority value for SCHED_RR is %d\n",sched_get_priority_max(SCHED_RR));
}

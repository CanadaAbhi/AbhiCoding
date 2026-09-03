/***************************************************************************
                          fork.c  -  description
                             -------------------

	used system calls are :
 	      pid_t fork(void);
 	      ssize_t write(int fd, const void *buf, size_t count);
    	      int creat(const char *pathname, mode_t mode);
       	      void _exit(int status);

	structure used : NOne
	Discription:
		Parent process create a file, by default this opens file for 
		write mode. Then creates child process. Since child gets 
		duplicate of the parent's file descriptor file remains 
		opend for child also. Both process writes into file in 
		co-operative way.(See manual for dup system call)

 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/
# include <stdio.h>
# include <stdlib.h>
 main()
 {
 	int pid,fd;
	fd = creat("newfile",0);
	pid = fork();
	printf("Hai");
	if(pid == 0)
	{
		write(fd," Child Process " ,17);
		printf(" Child process message printed\n");		
		printf("pid is %d\n",getpid());
		exit(0);
	}
	else
	{
		sleep(50);
		write(fd,"Parent Process ",17);
		printf(" Parent process message printed\n");
		printf("pid is %d\n",getpid());
	}
}			 
